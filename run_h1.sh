#!/usr/bin/env bash
# H1 ablation: both arms, same task, same seed. Only `return_mode` differs.
#   ./run_h1.sh                 -> smoke suite (tinycsv)
#   ./run_h1.sh jsonlines nl2repo
set -u
TASK="${1:-tinycsv}"
BENCH="${2:-smoke}"
MODEL="${MODEL:-deepseek/deepseek-chat}"
SEEDS="${SEEDS:-0}"

ARGS="--benchmark $BENCH --tasks $TASK
  --backend openrouter --model $MODEL
  --temperature 0.3 --max-output-tokens 8000
  --rpm-limit 60 --max-retries 4 --request-timeout-s 300
  --max-depth 1 --max-children 3 --max-concurrent-children 3
  --node-max-iterations 6 --root-max-iterations 8
  --max-wallclock-s 2400 --eval --traceback"
[ "$BENCH" = "nl2repo" ] && ARGS="$ARGS --nl2repo-root ${NL2REPO_ROOT:-../NL2RepoBench}"

for ARM in h1-bounded h1-full-source; do
  echo "===== $ARM / $TASK ($(date +%H:%M:%S)) ====="
  python -m rlmc.cli run --arm "$ARM" --seeds $SEEDS $ARGS
done
echo "===== REPORT ====="
python -m rlmc.cli report --compare h1-bounded h1-full-source --json runs/report.json
