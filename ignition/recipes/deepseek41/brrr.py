# SPDX-License-Identifier: Apache-2.0
"""TP3 experiment: BRRR's fused DS41 layer kernel (o-projection through MLP all-reduce) inside Flare forwards.

``TP3Config.brrr_layers`` selects layers. For those layers the decode and verify forwards replace inverse RoPE, the
grouped o-projection, the attention all-reduce, the FFN mHC, the router, routed and shared experts and the MLP
all-reduce with one kernel launch. Prefill keeps Flare's path.

The kernel (brrr/kernels/decoder.cu, Apache-2.0, fraserprice/DeepSeek-4.1-Flash-1017-BRRR, TP3 port) reads:
- routed experts from b12x's prepared W4A8 planes, scales decoded in-kernel from b12x's CSF payload
  (``brrr_native_scales`` loads these layers without CSF instead),
- dense weights as Flare's own row-major values (``brrr_dense_native``, default) or as its own staged copies read
  from the checkpoint view ``brrr_ckpt``; block scales always come from that view,
- peer flags/inboxes in Flare PeerStorage, so export resets them (and the epoch kept in the flag page) to zero.

Every switch arrives through the ``TP3Config`` the engine and the prepared tails hold; nothing here reads the
environment (``TORCH_EXTENSIONS_DIR`` is set for the extension build only).
"""

import ctypes as C
import json
import os
import sys
from functools import cache
from pathlib import Path

import torch
import torch.distributed as dist

from .tp3_config import TP3Config

_ROOT = Path(__file__).resolve().parents[3]


def _cuext():
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(_ROOT / ".brrr-ext"))
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from brrr import cuext

    return cuext


# Item-6 kernel switches: (TP3Config field, compile-time define, extension attribute). The split tail is a build switch
# (BRRR_SPLIT_TAIL=1) plus a per-engine count N (create(split_tail=N)), so one build serves every N.
SWITCHES = (("brrr_split_tail", "BRRR_SPLIT_TAIL", "SPLIT_TAIL"), ("brrr_fast_route", "BRRR_FAST_ROUTE", "FAST_ROUTE"),
            ("brrr_counter_parity", "BRRR_COUNTER_PARITY", "COUNTER_PARITY"), ("brrr_spin_trap", "BRRR_SPIN_TRAP", "SPIN_TRAP"))


def kernel_defines(tp3: TP3Config) -> tuple[str, ...]:
    """Compile-time defines of the layer kernel for this config (item-6 switches; all off = the plain build)."""
    return tuple(f"{define}=1" for field, define, _ in SWITCHES if getattr(tp3, field))


def engine_key(tp3: TP3Config):
    """What makes two engines differ: the build plus the per-engine split count."""
    return kernel_defines(tp3), int(tp3.brrr_split_tail)


@cache
def extension(defines: tuple[str, ...] = ()):
    """The layer kernel extension; ``defines`` (from kernel_defines) selects a compile-time variant."""
    return _cuext().extension("decoder", tuple(defines))


def check_switches(ext, tp3: TP3Config):
    """Contract check: the loaded build's compile-time switches are the ones the config asked for."""
    want = {attr: int(bool(getattr(tp3, field))) for field, _, attr in SWITCHES}
    got = {attr: int(getattr(ext, attr, -1)) for attr in want}
    if got != want:
        raise RuntimeError(f"BRRR kernel build switches {got} do not match the TP3 config {want}")


# Engines by global handle (the custom op's `handle`): (extension module, the module's own engine handle). Several
# engines (one per compile-time variant) can coexist in one process (the layer test's A/B pass); serving has one.
_RUNNERS = []


def _stages(w):
    w = w.contiguous().view(torch.uint8)
    w = torch.nn.functional.pad(w, (0, -w.shape[1] % 128))
    return w.reshape(w.shape[0] // 64, 4, 2, 8, -1, 4, 4, 8).permute(0, 4, 5, 1, 3, 6, 2, 7).contiguous().view(-1)


def _interleave(w, block):
    return w.reshape(2, -1, block, *w.shape[1:]).transpose(0, 1).reshape(w.shape).contiguous()


def _rows2d(weight):
    """Flare's row-major E4M3 values ([L*M, K]) from an MXFP8LinearWeight or MXFP8Rows, without copying."""
    rows = getattr(weight, "weight", weight)
    v = rows.values
    if v.ndim == 3:  # logical [M, K, L] view over physical [L, M, K]
        v = v.permute(2, 0, 1).reshape(-1, v.shape[1])
    if not v.is_contiguous() or v.data_ptr() != rows.values.data_ptr():
        raise RuntimeError("BRRR dense-native needs contiguous row-major values")
    return v


def _shard(rank, size):
    return slice(rank * size, (rank + 1) * size)


def check_constants(ext, config, weights):
    """Constants contract: the kernel's compile-time model constants against Flare's config and the mHC options
    decoder.py hands to PreparedMHC (rms_eps = norm_eps = rms_norm_eps, hc_eps, hc_sinkhorn_iters)."""
    import math

    text = config.text
    ints = {"HC": text["hc_mult"], "TOPK": text["num_experts_per_tok"], "ROPE": text["qk_rope_head_dim"],
            "EXPERTS": text["n_routed_experts"], "HID": text["hidden_size"], "SINKHORN": text["hc_sinkhorn_iters"]}
    floats = {"LIMIT": text["swiglu_limit"], "ROUTE_SCALE": text["routed_scaling_factor"],
              "HC_EPS": text["hc_eps"], "NORM_EPS": text["rms_norm_eps"]}
    bad = [f"{k}={getattr(ext, k)} (config {v})" for k, v in ints.items() if getattr(ext, k) != int(v)]
    bad += [f"{k}={getattr(ext, k)!r} (config {v!r})" for k, v in floats.items()
            if not math.isclose(getattr(ext, k), float(v), rel_tol=1e-6, abs_tol=0.0)]  # kernel floats are fp32
    if text.get("scoring_func") != "sqrtsoftplus" or not text.get("norm_topk_prob", False):
        bad.append(f"router scoring {text.get('scoring_func')!r}/norm_topk_prob={text.get('norm_topk_prob')!r} (kernel: sqrtsoftplus, normalized)")
    res = weights.residual
    if tuple(res["hc_ffn_fn"].shape) != (ext.MIX, ext.HC * ext.HID) or res["hc_ffn_base"].numel() != ext.MIX or res["hc_ffn_scale"].numel() != 3:
        bad.append(f"mHC fn {tuple(res['hc_ffn_fn'].shape)} base {res['hc_ffn_base'].numel()} scale {res['hc_ffn_scale'].numel()} "
                   f"(kernel MIX={ext.MIX}, HC*HID={ext.HC * ext.HID}, 3)")
    if bad:
        raise RuntimeError("BRRR kernel constants do not match the model: " + "; ".join(bad))


class Checkpoint:
    def __init__(self, root):
        from safetensors import safe_open

        self.root = Path(root)
        self.index = json.loads((self.root / "model.safetensors.index.json").read_text())["weight_map"]
        self._open = cache(lambda name: safe_open(self.root / name, framework="pt"))

    def shape(self, name):
        return self._open(self.index[name]).get_slice(name).get_shape()

    def get(self, name, rows=slice(None), cols=slice(None)):
        tensor = self._open(self.index[name]).get_slice(name)
        picked = tensor[rows, cols] if len(tensor.get_shape()) == 2 else tensor[rows]
        return picked.contiguous()


class Engine:
    """One per rank: the kernel's peer buffers, scratch and registered layers, configured by one TP3Config."""

    def __init__(self, group, device, ipc, tp3: TP3Config):
        if tp3.brrr_ckpt is None:
            raise ValueError("TP3 config: BRRR needs brrr_ckpt (the checkpoint view it reads dense copies from)")
        self.ext = ext = extension(kernel_defines(tp3))
        check_switches(ext, tp3)
        self.group, self.device, self.ipc, self.tp3 = group, device, ipc, tp3
        self.rank, world = dist.get_rank(group=group), dist.get_world_size(group=group)
        if world != ext.RANKS:
            raise ValueError(f"BRRR kernel is built for {ext.RANKS} ranks, group has {world}")
        # Same kernel build on every rank (handbook D.3.4); the item-6 switches only change rank-local work, but a
        # mixed build is a configuration error that would also invalidate any timing.
        builds = [None] * world
        dist.all_gather_object(builds, engine_key(tp3), group=group)
        if any(b != builds[0] for b in builds):
            raise RuntimeError(f"BRRR kernel switches (build, split count) differ between ranks: {builds}")
        nbytes = ext.flag_bytes() + ext.inbox_bytes()
        own = ipc.cudaMalloc(nbytes)
        ipc.storage.cuda.call("cuMemsetD8_v2", C.c_uint64(own), C.c_ubyte(0), C.c_size_t(nbytes))
        torch.cuda.synchronize(device)
        handles = [None] * world
        dist.all_gather_object(handles, ipc.cudaIpcGetMemHandleBytes(own), group=group)
        buffers = [own if r == self.rank else ipc.cudaIpcOpenMemHandleBytes(handles[r]) for r in range(world)]
        # The epoch counter sits in the unused tail of this rank's flag page (flags use 2*2*RANKS*CHUNKS words).
        # fused_attn_per_chunk is a per-engine kernel setting (every rank must agree: it shapes the flag protocol).
        local = ext.create(buffers, self.rank, epoch_addr=own + ext.flag_bytes() - 64,
                           attn_per_chunk_flags=int(tp3.fused_attn_per_chunk), split_tail=int(tp3.brrr_split_tail))
        self.handle = len(_RUNNERS)  # global handle (the custom op argument); serving has one engine, handle 0
        _RUNNERS.append((ext, local))
        self.local_handle = local
        dist.barrier(group=group)
        self.checkpoint = Checkpoint(tp3.brrr_ckpt)
        self.frequencies = None
        self.slots = {}
        self.keep = []

    def zero_ranges(self):
        """(address, bytes) ranges the export zeroes (export_zero_ranges). With brrr_counter_parity the counter
        allocation (both sets + the generation word) keeps the last launch's set; that state is consistent (the
        generation selects the clean set, independent of the flag-page epoch), so zeroing is not needed for
        correctness. It makes the package bytes independent of how many launches preparation ran and starts serving
        from both sets and the generation at zero. The plain build's counters are zero after every launch: nothing is
        added (package unchanged)."""
        if not self.tp3.brrr_counter_parity:
            return ()
        address, nbytes = self.ext.counters_range(self.local_handle)
        return ((int(address), int(nbytes)),)

    def raw(self, name, rows=slice(None), cols=slice(None)):
        return self.checkpoint.get(name, rows, cols).to(self.device)

    def raw_u8(self, name, rows=slice(None), cols=slice(None)):
        return self.raw(name, rows, cols).view(torch.uint8)

    def raw_padded(self, name, rows=slice(None), cols=slice(None)):
        """Rank shards may run past the 8 real o-groups (TP3 pads to 9); zero-fill like Flare's padded loader."""
        shape = self.checkpoint.shape(name)
        r0, r1 = rows.start or 0, rows.stop if rows.stop is not None else shape[0]
        if len(shape) == 2:
            c0, c1 = cols.start or 0, cols.stop if cols.stop is not None else shape[1]
            got = self.raw_u8(name, slice(r0, max(min(r1, shape[0]), r0)), slice(c0, max(min(c1, shape[1]), c0)))
            out = torch.zeros((r1 - r0, c1 - c0), dtype=torch.uint8, device=self.device)
            out[: got.shape[0], : got.shape[1]] = got
        else:
            got = self.raw_u8(name, slice(r0, max(min(r1, shape[0]), r0)))
            out = torch.zeros((r1 - r0,), dtype=torch.uint8, device=self.device)
            out[: got.shape[0]] = got
        return out

    def register(self, index, weights, config=None):
        if index in self.slots:
            return self.slots[index]
        ext, r, p, tp3 = self.ext, self.rank, f"layers.{index}", self.tp3
        if config is not None:
            check_constants(ext, config, weights)
        inter, oa = ext.INTER, ext.OA
        experts = weights.experts.experts
        impl = experts._impl
        rep = impl.representation_for("w4a8_mx")
        if rep is None or rep.n64_repack or rep.intermediate_size != inter or rep.num_experts != ext.EXPERTS:
            raise RuntimeError(f"BRRR layer {index}: unexpected expert representation {rep}")
        u8 = lambda t: t.contiguous().view(torch.uint8)
        rows = _shard(r, inter)
        gate_up = None
        if not tp3.brrr_dense_native or tp3.brrr_csf_check:
            gate_up = torch.cat((self.raw(f"{p}.ffn.shared_experts.w1.weight", rows),
                                 self.raw(f"{p}.ffn.shared_experts.w3.weight", rows)))
        blocks = _shard(r, inter // 32)
        scales = (self.raw_u8(f"{p}.ffn.shared_experts.w1.scale", blocks), self.raw_u8(f"{p}.ffn.shared_experts.w3.scale", blocks))
        kwargs = dict(
            woa_s=u8(self.raw_padded(f"{p}.attn.wo_a.scale", _shard(r, oa // 32))),
            wob_s=u8(self.raw_padded(f"{p}.attn.wo_b.scale", cols=_shard(r, oa // 32))),
            gate_b=self.raw(f"{p}.ffn.gate.bias").float().contiguous(),
            bias_vl=None,
            sh13_s=_interleave(torch.cat(scales), 1),
            sh2_s=torch.nn.functional.pad(self.raw_u8(f"{p}.ffn.shared_experts.w2.scale", cols=blocks),
                                          (0, ext.SH2_LD - inter // 32)).contiguous(),
            w13=rep.w13_rp.contiguous(), s13=rep.w13_sfb.contiguous(),
            w2=rep.w2_rp.contiguous(), s2=rep.w2_sfb.contiguous(),
            fn=self.raw(f"{p}.hc_ffn_fn").float().contiguous(),
            hc_scale=self.raw(f"{p}.hc_ffn_scale").float().contiguous(),
            hc_base=self.raw(f"{p}.hc_ffn_base").float().contiguous(),
            norm=self.raw(f"{p}.ffn_norm.weight").contiguous(),
            image_lo=0, dense_native=False,
            # Merged hot/cold bank: experts are stored hot-first; the kernel routes by global ID and reads the slot.
            hot_map=None if weights.experts.hot_map is None else weights.experts.hot_map.to(torch.int32).contiguous(),
            # Cold-first job ordering: slots >= hot_rows are host-RAM experts. Default EXPERTS (all resident) keeps
            # the arrival order; a merged bank without a known hot count also stays unordered (safe).
            hot_rows=ext.EXPERTS if weights.experts.hot_map is None or weights.experts.hot_rows is None
                     else int(weights.experts.hot_rows))
        if tp3.brrr_dense_native:
            # Read Flare's own row-major E4M3 dense values (same bytes as the checkpoint shards; block scales stay as
            # small checkpoint copies) instead of staged copies: ~35-40 MB per layer back to the KV pool.
            out, shared = weights.attention["output"], weights.experts.shared
            native = dict(woa=_rows2d(out.wo_a), wob=_rows2d(out.wo_b), gate=weights.experts.router,
                          sh13=_rows2d(shared.gate_up), sh2=_rows2d(shared.down))
            if tp3.brrr_csf_check:
                ref = dict(woa=self.raw_padded(f"{p}.attn.wo_a.weight", _shard(r, oa)),
                           wob=self.raw_padded(f"{p}.attn.wo_b.weight", cols=_shard(r, oa)),
                           gate=self.raw_u8(f"{p}.ffn.gate.weight"), sh13=u8(gate_up),
                           sh2=self.raw_u8(f"{p}.ffn.shared_experts.w2.weight", cols=rows))
                for name, t in native.items():
                    a, b = t.contiguous().view(torch.uint8), ref[name]
                    same = a.shape == b.shape and bool(torch.equal(a, b))
                    print(f"[brrr] rank {r} layer {index} native {name} {tuple(t.shape)} {t.dtype} "
                          f"stride {tuple(t.stride())}: {'identical' if same else f'MISMATCH vs {tuple(b.shape)}'}", flush=True)
                    if not same:
                        raise RuntimeError(f"BRRR layer {index}: Flare's {name} differs from the checkpoint bytes")
            kwargs.update(native, dense_native=True)
        else:
            kwargs.update(woa=_stages(self.raw_padded(f"{p}.attn.wo_a.weight", _shard(r, oa))),
                          wob=_stages(self.raw_padded(f"{p}.attn.wo_b.weight", cols=_shard(r, oa))),
                          gate=_stages(self.raw(f"{p}.ffn.gate.weight")), sh13=_stages(_interleave(gate_up, 32)),
                          sh2=_stages(self.raw(f"{p}.ffn.shared_experts.w2.weight", cols=rows)))
        decoder = impl.nvfp4_csf
        if decoder is not None:
            # Same b12x CSF payload as the vLLM build BRRR's in-kernel reader was qualified on.
            for tag, plane in (("csf13", decoder.first), ("csf2", decoder.second)):
                a = _csf_args(plane.scales)
                kwargs[f"{tag}_storage"], kwargs[f"{tag}_gdesc"], kwargs[f"{tag}_meta"] = a["storage"], a["gdesc"], a["meta"]
            if tp3.brrr_csf_check:
                decoder.decode_all(impl.w1_blockscale, impl.w2_blockscale)
                for tag, ref in (("csf13", rep.w13_sfb), ("csf2", rep.w2_sfb)):
                    want = ref.contiguous().view(torch.int32).reshape(-1)
                    got = torch.empty_like(want)
                    ext.csf_expand_all(kwargs[f"{tag}_storage"], kwargs[f"{tag}_gdesc"], kwargs[f"{tag}_meta"], got)
                    torch.cuda.synchronize(self.device)
                    print(f"[brrr] rank {r} layer {index} {tag} CSF self-check: {int((got != want).sum())} of "
                          f"{want.numel()} words differ", flush=True)
        for name in ("w13", "s13", "w2", "s2"):
            if kwargs[name].data_ptr() != getattr(rep, {"w13": "w13_rp", "s13": "w13_sfb", "w2": "w2_rp", "s2": "w2_sfb"}[name]).data_ptr():
                raise RuntimeError(f"BRRR layer {index}: {name} is not b12x's resident plane")
        self.keep.append(kwargs)  # the kernel holds raw pointers to these tensors
        self.slots[index] = ext.add_layer(self.local_handle, **kwargs)
        if tp3.brrr_log:
            print(f"[brrr] rank {r} layer {index} -> slot {self.slots[index]} (w13 {tuple(rep.w13_rp.shape)}, "
                  f"sfb {tuple(rep.w13_sfb.shape)}, layout {rep.w13_layout})", flush=True)
        return self.slots[index]


def _csf_args(scales):
    """Describe a b12x CsfBlockPayload (destination 2, mxfp4, 3-byte group records) for the fused kernel: per-(payload
    expert, group) record descriptors first | count << 22, mirroring BlockStorageLayout (from BRRR's TP3 decoder.py)."""
    st = scales.storage
    R, C, Es = int(scales.rows), int(scales.columns), int(scales.num_experts)
    if not (scales.destination == 2 and scales.native_codes and scales.group_record_bytes == 3 and scales.quantization == "mxfp4"):
        raise RuntimeError(f"BRRR kernel needs CSF destination 2 / mxfp4 / 3-byte records; got {scales}")
    a16 = lambda v: (v + 15) // 16 * 16
    cg, groups = (C + 3) // 4, (R + 127) // 128 * ((C + 3) // 4)
    pages = (groups + 31) // 32
    prefixes, code_bytes = groups + pages, R * C // 8
    bases = 1024
    codes = a16(bases + Es * R)
    page_bases = a16(codes + Es * code_bytes)
    offsets = a16(page_bases + Es * pages * 4)
    recpos = a16(offsets + Es * prefixes * 2)
    value_off = int(st[:8].view(torch.int64).item())
    pb = st[page_bases:page_bases + Es * pages * 4].view(torch.int32).view(Es, pages)
    off = st[offsets:offsets + Es * prefixes * 2].view(torch.int16).view(Es, prefixes).to(torch.int32) & 0xFFFF
    sg = torch.arange(groups, device=st.device)
    page = sg // 32
    first = pb[:, page] + off[:, sg + page]
    count = pb[:, page] + off[:, sg + page + 1] - first
    if int(first.max()) >= 1 << 22 or int(count.max()) >= 1 << 10 or int(count.min()) < 0:
        raise RuntimeError(f"CSF record descriptor out of range: first<{int(first.max())} count<{int(count.max())}")
    gdesc = (first | (count << 22)).to(torch.int32).contiguous()
    return dict(storage=st, gdesc=gdesc, meta=[bases, codes, code_bytes, value_off, recpos, R, cg, groups, int(scales.sections)])


_ENGINE = []


def create_engine(group, device, ipc, tp3: TP3Config):
    """Create the rank's engine once; None (no-op) unless ``tp3.brrr_layers`` selects layers."""
    if not tp3.brrr_layers:
        return None
    if not _ENGINE:
        _ENGINE.append(Engine(group, device, ipc, tp3))
    return _ENGINE[0]


def engine():
    return _ENGINE[0] if _ENGINE else None


@torch.library.custom_op("flare::brrr_layer_tail",
                         mutates_args=("attn_post", "attn_comb", "attn_pre", "y", "resid1", "post1", "comb1", "pre1"))
def brrr_layer_tail(handle: int, slot: int, o: torch.Tensor, positions: torch.Tensor, frequencies: torch.Tensor,
                    resid0: torch.Tensor, attn_post: torch.Tensor, attn_comb: torch.Tensor, attn_pre: torch.Tensor,
                    y: torch.Tensor, resid1: torch.Tensor, post1: torch.Tensor, comb1: torch.Tensor,
                    pre1: torch.Tensor) -> None:
    ext, local = _RUNNERS[handle]
    ext.run(local, slot, o, positions, None, frequencies, resid0, attn_post, attn_comb, attn_pre,
            y, resid1, post1, comb1, pre1)


@brrr_layer_tail.register_fake
def _fake_brrr_layer_tail(handle, slot, o, positions, frequencies, resid0, attn_post, attn_comb, attn_pre,
                          y, resid1, post1, comb1, pre1):
    pass


@cache
def sparse_extension():
    return _cuext().extension("flare_sparse_decode")


@torch.library.custom_op("flare::brrr_sparse_decode", mutates_args=("out", "partial", "stats", "counters"))
def brrr_sparse_decode(q: torch.Tensor, out: torch.Tensor, swa: torch.Tensor, swa_page: int, swa_indices: torch.Tensor,
                       swa_lengths: torch.Tensor, idx: torch.Tensor | None, idx_page: int,
                       idx_logical: torch.Tensor | None, idx_lengths: torch.Tensor | None,
                       idx_table: torch.Tensor | None, sink: torch.Tensor, scale: float, partial: torch.Tensor,
                       stats: torch.Tensor, counters: torch.Tensor) -> None:
    sparse_extension().decode(q, out, swa, swa_page, swa_indices, swa_lengths, idx, idx_page, idx_logical,
                              idx_lengths, idx_table, sink, scale, partial, stats, counters)


@brrr_sparse_decode.register_fake
def _fake_brrr_sparse_decode(q, out, swa, swa_page, swa_indices, swa_lengths, idx, idx_page, idx_logical,
                             idx_lengths, idx_table, sink, scale, partial, stats, counters):
    pass


SPARSE_ROWS = 64  # shared split-K scratch capacity; wider graphs keep b12x attention
_SPARSE_SCRATCH = {}


def sparse_supported(rows, swa_width, idx_width, heads, dim) -> bool:
    ext = sparse_extension()
    return (rows <= SPARSE_ROWS and heads == ext.H and dim == ext.D and swa_width == 128
            and ext.ctas_for(swa_width + idx_width) > 0)


class PreparedBRRRSparse:
    """Sparse MLA decode (window + indexed keys, attention sink) in one BRRR launch per layer: per-CTA key splits
    with the softmax merge done by the last CTA, instead of b12x's decode + merge kernels (TP3Config.brrr_sparse).

    MEASURED 2026-11: level with the stock kernels at c1/c8 -> not adopted; kept opt-in with its kernel test
    (tp3-tools/test_flare_sparse_decode.py)."""

    def __init__(self, query, window, window_page_size, window_indices, window_lengths, sink, output, *,
                 indexed=None, indexed_page_size=0, indexed_indices=None, indexed_lengths=None, page_table=None):
        ext = sparse_extension()
        device = query.device
        key = device.index
        if key not in _SPARSE_SCRATCH:
            _SPARSE_SCRATCH[key] = (torch.zeros(SPARSE_ROWS * ext.CL * ext.H * ext.D, device=device),
                                    torch.zeros(SPARSE_ROWS * ext.CL * ext.H * 2, device=device),
                                    torch.zeros(SPARSE_ROWS, device=device, dtype=torch.int32))
        self.partial, self.stats, self.counters = _SPARSE_SCRATCH[key]
        if page_table is not None and page_table.dtype != torch.int32:
            raise ValueError("BRRR sparse decode expects an int32 page table")
        self.query, self.window, self.window_page_size = query, window, int(window_page_size)
        self.window_indices, self.window_lengths, self.output = window_indices, window_lengths, output
        self.sink = sink if sink.dtype == torch.float32 else sink.float().contiguous()
        self.indexed, self.indexed_page_size = indexed, int(indexed_page_size)
        self.indexed_indices, self.indexed_lengths, self.page_table = indexed_indices, indexed_lengths, page_table
        self.scale = query.shape[-1] ** -.5

    def __call__(self):
        brrr_sparse_decode(self.query, self.output, self.window, self.window_page_size, self.window_indices,
                           self.window_lengths, self.indexed, self.indexed_page_size, self.indexed_indices,
                           self.indexed_lengths, self.page_table, self.sink, self.scale, self.partial, self.stats,
                           self.counters)
        return self.output


class PreparedBRRRTail:
    """Layer tail through the BRRR kernel. Reads s.attended (pre-inverse-RoPE attention output)."""

    def __init__(self, index, weights, scratch, positions, frequencies, config=None, tp3=None, engine_=None):
        e = engine() if engine_ is None else engine_  # engine_: an explicit engine (test A/B of kernel variants)
        if e is None:
            raise RuntimeError("BRRR engine was not created")
        tp3 = e.tp3 if tp3 is None else tp3
        self.handle, self.slot = int(e.handle), int(e.register(index, weights, config))
        self.scratch, self.positions, self.frequencies = scratch, positions, frequencies
        if tp3.brrr_break == "rope":
            # Negative control for test tooling: reversed RoPE frequencies (wrong inverse rotation of the attention output).
            self.frequencies = frequencies.flip(0).contiguous()
        if positions.dtype != torch.int64:
            raise ValueError("BRRR tail expects int64 positions")

    def __call__(self, attended):
        s = self.scratch
        brrr_layer_tail(self.handle, self.slot, attended, self.positions, self.frequencies,
                        s.attention_hc.residual, s.attention_hc.post, s.attention_hc.comb, s.attn_mix,
                        s.mlp_output, s.ffn_hc.residual, s.ffn_hc.post, s.ffn_hc.comb, s.pre_mix)
