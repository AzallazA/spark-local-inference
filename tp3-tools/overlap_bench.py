"""Microbenchmark for TP3 hot/cold expert overlap on one parked layer.

Cold experts live in real pinned, device-mapped host memory. Routes are sampled from the
vLLM route counts. Each case is captured as a CUDA graph two ways: serial (hot then cold
on one stream) and forked (cold on a second stream, joined before the sum).
usage: overlap_bench.py SNAPSHOT LAYER [HOT=326]
"""
import json, os, sys
import torch

from b12x.preparation import PreparationSession
from b12x.sequence._shared.disk_table import _tensor_from_pointer
from ignition.components.moe import PreparedRoutedExperts
from ignition.components.parallel import TensorParallelShard
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.recipes.deepseek41.weights import DeepSeekWeightLoader

snap, L = sys.argv[1], int(sys.argv[2])
H = int(sys.argv[3]) if len(sys.argv) > 3 else 326
dev = torch.device("cuda", 0); torch.cuda.set_device(dev)
shard = TensorParallelShard(0, 3)
ffn = f"layers.{L}.ffn"
counts_dir = "~/ds41-tp3/release/ds41-kraken-tp3-20261019/expert-order-blend"
raw = json.load(open(f"{counts_dir}/expert-routes-rank0-layer{L}.json"))
counts = raw.get("counts", raw) if isinstance(raw, dict) else raw
if isinstance(counts, dict):
    counts = [counts[k] for k in sorted(counts, key=int)]
counts = torch.tensor(counts, dtype=torch.float64)
order = torch.argsort(counts, descending=True)
hot, cold = sorted(order[:H].tolist()), sorted(order[H:].tolist())
pinned = []


def host_alloc(name, shape, dtype):
    if os.environ.get("COLD_ON_DEVICE"):
        view = torch.empty(shape, dtype=dtype, device=dev); pinned.append(view)

        class Region:
            device_view = view
        return Region
    # Real mapped host memory: pinned allocation, device view at the same UVA address.
    host = torch.empty(shape, dtype=dtype, pin_memory=True)
    pinned.append(host)
    view = _tensor_from_pointer(host.data_ptr(), shape=tuple(shape), dtype=dtype, device=dev, nbytes=host.nbytes)

    class Region:
        device_view = view
    return Region


with torch.no_grad(), DeepSeekWeightLoader(inspect_indexed_checkpoint(snap), dev, shard=shard,
                                           expert_scale_compression="csf", prefixes=(f"{ffn}.experts.",)) as loader:
    text = loader.config.text
    E = text["n_routed_experts"]
    interval = shard.partition(text["moe_intermediate_size"], alignment=32)
    hot_bank = loader.load_experts([f"{ffn}.experts.{i}" for i in hot], interval)
    loader._host_allocator = host_alloc
    cold_bank = loader.load_experts([f"{ffn}.experts.{i}" for i in cold], interval, host=True)
    sys.path.insert(0, "~/flare-tp3-tools")
    import mixed_bank
    loader._host_allocator = mixed_bank.allocator(H, dev)
    merged_bank = loader.load_experts([f"{ffn}.experts.{i}" for i in hot + cold], interval, host=True)
    loader._host_allocator = None
    device_bank = loader.load_experts([f"{ffn}.experts.{i}" for i in hot + cold], interval)
hot_map = torch.full((E,), -1, dtype=torch.int32); hot_map[hot] = torch.arange(len(hot), dtype=torch.int32)
cold_map = torch.full((E,), -1, dtype=torch.int32); cold_map[cold] = torch.arange(len(cold), dtype=torch.int32)
perm = torch.empty(E, dtype=torch.int32); perm[torch.tensor(hot + cold)] = torch.arange(E, dtype=torch.int32)
hot_map, cold_map, perm = hot_map.to(dev), cold_map.to(dev), perm.to(dev)
prob = counts / counts.sum()
print(f"layer {L}: {H} hot / {E - H} cold (host), cold route share {prob[cold].sum() * 100:.2f}%")

hidden = text["hidden_size"]
aux = torch.cuda.Stream()
flush = torch.empty(512 << 20, dtype=torch.uint8, device=dev)
with torch.no_grad(), PreparationSession(device=dev, autotune=False, compile_workers=0) as session:
    for rows in (6, 24, 48):
        x = torch.randn((rows, hidden), dtype=torch.bfloat16, device=dev)
        ids = torch.multinomial(prob, rows * 6, replacement=True).view(rows, 6).int().to(dev)
        w = torch.rand((rows, 6), dtype=torch.float32, device=dev)
        hid, cid = torch.empty_like(ids), torch.empty_like(ids)
        torch.index_select(hot_map, 0, ids.view(-1), out=hid.view(-1))
        torch.index_select(cold_map, 0, ids.view(-1), out=cid.view(-1))
        out_hot, out_cold, out_merged = torch.zeros_like(x), torch.zeros_like(x), torch.zeros_like(x)
        mid = torch.empty_like(ids); torch.index_select(perm, 0, ids.view(-1), out=mid.view(-1))
        op_hot = PreparedRoutedExperts(x, hot_bank, hid, w, out_hot, session=session, name=f"hot{rows}")
        op_cold = PreparedRoutedExperts(x, cold_bank, cid, w, out_cold, session=session, name=f"cold{rows}")
        op_merged = PreparedRoutedExperts(x, merged_bank, mid, w, out_merged, session=session, name=f"merged{rows}")
        out_device = torch.zeros_like(x)
        op_device = PreparedRoutedExperts(x, device_bank, mid, w, out_device, session=session, name=f"device{rows}")
        n_cold = int((cid >= 0).sum()); n_cold_experts = int(torch.unique(cid[cid >= 0]).numel())

        def serial():
            op_hot(); op_cold(); out_hot.add_(out_cold)

        def forked():
            aux.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(aux):
                op_cold()
            op_hot()
            torch.cuda.current_stream().wait_stream(aux)
            out_hot.add_(out_cold)

        def hot_only():
            op_hot()

        def merged():
            op_merged()

        results = {}
        for label, fn in (("hot only", hot_only), ("serial", serial), ("forked", forked), ("merged", merged)):
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.graph(g, stream=s):
                fn()
            torch.cuda.synchronize()
            for _ in range(20):
                g.replay()
            # Flush the 128 MB L2 between replays (real decode streams ~40 layers through L2 between visits).
            times = []
            for _ in range(100):
                flush.fill_(1)
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record(); g.replay(); end.record()
                torch.cuda.synchronize(); times.append(start.elapsed_time(end) * 1000)
            times.sort(); results[label] = times[len(times) // 2]
        serial()
        merged(); torch.cuda.synchronize()
        err = float((out_merged.float() - out_hot.float()).norm() / out_hot.float().norm().clamp_min(1e-9))
        op_device(); torch.cuda.synchronize()
        exact = torch.equal(out_merged, out_device)
        rel = lambda a, b: float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-9))
        cold_rows = (cid >= 0).any(-1)
        print(f"    merged vs all-VRAM: {'bit-identical' if exact else 'DIFFERS'} rel {rel(out_merged, out_device):.5f}"
              f" | all-VRAM vs serial rel {rel(out_device, out_hot):.5f}"
              f" | rows with a cold route: merged-vs-VRAM rel {rel(out_merged[cold_rows], out_device[cold_rows]) if cold_rows.any() else float('nan'):.5f}"
              f" | other rows {rel(out_merged[~cold_rows], out_device[~cold_rows]):.5f}")
        first = out_device.clone(); op_device(); torch.cuda.synchronize()
        print(f"    all-VRAM vs itself (rerun): rel {rel(out_device, first):.5f}")
        out_device.copy_(first); out_merged.zero_()
        for k in range(rows):   # zero all but one cold route to see whether host-backed weights contribute
            pass
        a = merged_bank.w13 if hasattr(merged_bank, "w13") else None
        saved = results["serial"] - results["merged"]
        print(f"  rows {rows:2d} ({n_cold} cold routes, {n_cold_experts} cold experts): hot only {results['hot only']:6.1f} us"
              f" | serial {results['serial']:6.1f} us | forked {results['forked']:6.1f} us"
              f" | merged {results['merged']:6.1f} us | merged saves {saved:5.1f} us/layer ({saved / results['serial'] * 100:4.1f}%) | rel err {err:.4f}")
