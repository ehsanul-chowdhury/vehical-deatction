#!/usr/bin/env bash
# Poll the Kaggle run; on completion pull weights into ./weights.
#   ./watch_kaggle.sh <kaggle-username> [15class|10class]
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
KAGGLE="$ROOT/.venv/bin/kaggle"
USER_NAME="${1:?usage: $0 <kaggle-username> [15class|10class]}"
MODE="${2:-15class}"
case "$MODE" in
  15class) SLUG="eee-project-yolov9-15class" ;;
  10class) SLUG="eee-project-yolov9-10class" ;;
  *) echo "unknown mode: $MODE" >&2; exit 2 ;;
esac
REF="$USER_NAME/$SLUG"

while true; do
  S="$("$KAGGLE" kernels status "$REF" 2>&1 | tr -d '\n')"
  echo "$(date +%H:%M:%S)  $S"
  case "$S" in
    *complete*)
      mkdir -p "$ROOT/models"
      "$KAGGLE" kernels output "$REF" -p "$ROOT/models"
      echo "--- downloaded ---"; ls -lh "$ROOT/models" | grep -E "\.pt|\.onnx|classes|\.jpg" || true
      exit 0 ;;
    *error*|*cancel*)
      echo "RUN FAILED -- fetching log"
      "$KAGGLE" kernels output "$REF" -p "$ROOT/models" 2>/dev/null || true
      exit 1 ;;
  esac
  sleep 120
done
