#!/bin/bash
# After run_all.sh: the same TriviaQA + MMLU-Pro on the Flare default build (MXFP4 Engram in RAM). Flare clamps
# temperature to 1.0, so each benchmark runs twice to show its sampling spread.
cd $HOME/engram-eval
until grep -q "engram eval done" run_all.log; do sleep 20; done
for i in $(seq 1 120); do curl -s -m 2 localhost:15015/v1/models | grep -q '"id"' && break; sleep 5; done
echo "=== $(date +%H:%M) Flare: $(ps -eo args | grep 'release/[f]lare run' | grep -o 'gb202[^ ]*')"
for r in flare1 flare2; do ENGINE=flare .venv/bin/python run_eval.py triviaqa $r; ENGINE=flare .venv/bin/python run_eval.py mmlupro $r; done
echo "=== flare runs done"
