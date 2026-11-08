# Engram precision evaluation (FP8 vs MXFP4 tables)

Tooling behind `docs/tp3/ENGRAM-PRECISION.md`: teacher-forced perplexity on Wikipedia, code (this repository's own
`ignition/` and `helios/` trees) and TriviaQA/MMLU-Pro answer sets, run against a vLLM server (FP8 tables) and a
Flare server (MXFP4 tables), then compared.

| File | Role |
| --- | --- |
| `build_sets.py` | builds the passage and question sets (`passages.jsonl`, `triviaqa.jsonl`, `mmlupro.jsonl`) |
| `run_eval.py` | one server, one set: prompt logprobs one passage at a time (several at once OOM a vLLM engine), answers with up to 2,048 output tokens |
| `compare.py` | side-by-side perplexity and accuracy tables |
| `run_all.sh`, `run_flare.sh`, `prefix_chain.sh` | site-specific drivers (server start/stop, paths under `$HOME`); edit before use |

Results and method: `docs/tp3/ENGRAM-PRECISION.md`.
