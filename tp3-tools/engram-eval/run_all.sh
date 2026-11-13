#!/bin/bash
# FP8 vs MXFP4 Engram on vLLM Kraken TP3 (.11): stop Flare, serve stock FP8 (layer 1 table in RAM, layer 14 from disk),
# run logprobs + TriviaQA + MMLU-Pro twice (noise floor), then the engram-nvidia MXFP4 variant (both tables in RAM),
# compare, restore the vLLM .env and the Flare default server.
set -u
E=$HOME/engram-eval; R=$HOME/ds41-tp3/release/ds41-kraken-tp3-20261019
cd $E
pids(){ ps -eo pid,args | awk '$2 ~ /(^|\/)flare$/ && $3=="run"{print $1}'; }
gpu_free(){ until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ]; do sleep 3; done; }
ready(){ for i in $(seq 1 300); do curl -s -m 2 localhost:15015/v1/models | grep -q '"id"' && return 0; sleep 5; done; echo "=== vLLM not ready"; return 1; }
for p in $(pids); do kill -INT $p; done; for i in $(seq 1 60); do [ -z "$(pids)" ] && break; sleep 3; done; for p in $(pids); do kill -KILL $p; done
gpu_free
cp $R/.env $E/env.backup
serve(){ # <model> <ram layers>
  sed -i -E "s|^MODEL=.*|MODEL=$1|; s|^ENGRAM_RAM_LAYERS=.*|ENGRAM_RAM_LAYERS=$2|" $R/.env
  (cd $R && docker compose up -d --force-recreate > $E/compose.log 2>&1); ready; }
echo "=== $(date +%H:%M) FP8 (stock)"
serve deepseek-ai/DeepSeek-V4.1-Flash-1017 1 && {
  .venv/bin/python run_eval.py logprobs fp8a; .venv/bin/python run_eval.py triviaqa fp8a; .venv/bin/python run_eval.py mmlupro fp8a
  .venv/bin/python run_eval.py logprobs fp8b; .venv/bin/python run_eval.py triviaqa fp8b; .venv/bin/python run_eval.py mmlupro fp8b; }
echo "=== $(date +%H:%M) MXFP4 (engram-nvidia)"
serve local/DeepSeek-V4.1-Flash-1017-engram-nvidia 1,14 && {
  .venv/bin/python run_eval.py logprobs mxfp4; .venv/bin/python run_eval.py triviaqa mxfp4; .venv/bin/python run_eval.py mmlupro mxfp4; }
(cd $R && docker compose down > /dev/null 2>&1); cp $E/env.backup $R/.env
echo "=== noise floor (FP8 vs FP8)"; .venv/bin/python compare.py fp8a fp8b
echo "=== FP8 vs MXFP4"; .venv/bin/python compare.py fp8a mxfp4
gpu_free
nohup $HOME/flare-up/serve-best.sh > $HOME/flare-up-v3/serve-best.log 2>&1 &
for i in $(seq 1 120); do sleep 5; curl -s -m 2 localhost:15015/v1/models | grep -q '"id"' && break; done
echo "=== $(date +%H:%M) Flare default restored; engram eval done"
