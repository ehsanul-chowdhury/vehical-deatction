#!/usr/bin/env bash
# Push a notebook to Kaggle and start it on a T4.
#   ./push_to_kaggle.sh <kaggle-username> [15class|10class]
#
#   15class (default) - downloads the raw 7.48 GB VOC dataset, converts it,
#                        trains on all 15 classes (reproduces the paper)
#   10class            - trains on the ready-made 10-class pre-converted cut
#                        (the same data as archive.zip), mounted from Kaggle
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
KAGGLE="$ROOT/.venv/bin/kaggle"

USER_NAME="${1:-}"
MODE="${2:-15class}"
if [ -z "$USER_NAME" ]; then
  echo "usage: $0 <kaggle-username> [15class|10class]" >&2
  echo "(username from your profile URL: kaggle.com/<username>)" >&2
  exit 2
fi

# Kaggle rejects the push unless the title slugifies to the id, so TITLE and
# SLUG are kept in lockstep and the title carries no punctuation.
case "$MODE" in
  15class) NB="EEE_project_train_15class.ipynb"; SLUG="eee-project-yolov9-15class"
            TITLE="EEE project YOLOv9 15class"; SOURCES='' ;;
  10class) NB="EEE_project_train_10class.ipynb";  SLUG="eee-project-yolov9-10class"
            TITLE="EEE project YOLOv9 10class"; SOURCES='"ehsanulc/eee-project-vehicle-dataset"' ;;
  *) echo "unknown mode: $MODE (use 15class or 10class)" >&2; exit 2 ;;
esac

if ! "$KAGGLE" kernels list --mine >/dev/null 2>&1; then
  echo "Not authenticated. Run:  $KAGGLE auth login" >&2
  exit 3
fi

"$ROOT/.venv/bin/python" "$ROOT/training/build_notebook.py"

PUSH="$ROOT/.kaggle_push"
rm -rf "$PUSH"; mkdir -p "$PUSH"
cp "$ROOT/notebooks/$NB" "$PUSH/"

# enable_internet is required in both modes: ultralytics and the pretrained
# yolov9*.pt checkpoints are fetched at runtime (15class also pulls the dataset).
cat > "$PUSH/kernel-metadata.json" <<JSON
{
  "id": "${USER_NAME}/${SLUG}",
  "title": "${TITLE}",
  "code_file": "${NB}",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "machine_shape": "NvidiaTeslaT4",
  "dataset_sources": [${SOURCES}],
  "competition_sources": [],
  "kernel_sources": [],
  "model_sources": []
}
JSON

echo "--- pushing ${USER_NAME}/${SLUG} (${MODE}) ---"
"$KAGGLE" kernels push -p "$PUSH"
echo
echo "Running at: https://www.kaggle.com/code/${USER_NAME}/${SLUG}"
echo "Poll with:  $ROOT/training/watch_kaggle.sh ${USER_NAME} ${MODE}"
