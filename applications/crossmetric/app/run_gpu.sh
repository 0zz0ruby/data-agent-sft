#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash app/run_gpu.sh /path/to/fine-tuned-model [port]"
  exit 2
fi

CHECKPOINT_PATH="$1"
SERVER_PORT="${2:-6008}"

python -u app/crossmetric_app.py \
  --checkpoint-path "$CHECKPOINT_PATH" \
  --local-files-only \
  --server-name 127.0.0.1 \
  --server-port "$SERVER_PORT" \
  --max-new-tokens 768
