#!/usr/bin/env python3
"""Generate the Kaggle notebook from scripts/ so there is one source of truth.

Kaggle runs a single code file, so the converter scripts are embedded as
%%writefile cells rather than maintained twice.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "notebooks" / "EEE_project_train_15class.ipynb"


# nbformat joins a list-valued `source` with "" -- each element must therefore
# end in its own newline. Emitting a single string sidesteps that entirely and
# is equally valid; a list without trailing newlines silently collapses the whole
# cell onto one line, which is a SyntaxError only at run time.
_n = __import__("itertools").count()


def md(src: str) -> dict:
    return {"cell_type": "markdown", "id": f"md{next(_n)}",
            "metadata": {}, "source": src.strip()}


def code(src: str) -> dict:
    return {"cell_type": "code", "id": f"cd{next(_n)}", "execution_count": None,
            "metadata": {}, "outputs": [], "source": src.strip("\n")}


def writefile(name: str, path: Path) -> dict:
    body = path.read_text().rstrip("\n")
    return code(f"%%writefile {name}\n{body}")


cells = [
    md("""
# EEE_project -> YOLOv9 vehicle detection (15 classes)

Reproduces arXiv:2410.08230 (YOLOv9-C, mAP@0.5 = 0.934) on the BD vehicle dataset.

Pipeline: fetch raw VOC data -> audit -> convert to YOLO -> verify -> train -> export ONNX.

The raw 7.48 GB dataset is pulled straight from Mendeley (the Kaggle mirrors are
unusable: one is a 10-class cut missing boat/horsecart/launch/tractor/
wheelbarrow, the other is classification-only with no boxes). **Internet must be ON.**

Everything bulky lives in `/kaggle/temp` (ephemeral) and is deleted as soon as it
is consumed; only weights and reports land in `/kaggle/working`.
"""),
    code("""
import os, subprocess, sys, glob, shutil, zipfile, time
from pathlib import Path

gpu = subprocess.run(["nvidia-smi","--query-gpu=name,memory.total","--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip()
print("GPU:", gpu or "NONE -- enable the T4 accelerator in the right-hand panel")

TMP  = Path("/kaggle/temp"); TMP.mkdir(parents=True, exist_ok=True)
RAW  = TMP / "raw"
YOLO = TMP / "eee_project_yolo"
WORK = Path("/kaggle/working"); WORK.mkdir(exist_ok=True)
ZIP  = TMP / "eee_project_data.zip"
URL  = "https://data.mendeley.com/public-api/zip/pwyyg8zmk5/download/2"

def disk_free_gb():
    st = os.statvfs("/kaggle/temp"); return st.f_bavail * st.f_frsize / 1e9

if not RAW.exists():
    print(f"free disk: {disk_free_gb():.1f} GB -- downloading 7.48 GB ...", flush=True)
    t0 = time.time()
    rc = subprocess.run(["curl","-L","--fail","--retry","3","--retry-delay","5",
                         "-o", str(ZIP), URL]).returncode
    assert rc == 0, f"download failed rc={rc}"
    print(f"downloaded {ZIP.stat().st_size/1e9:.2f} GB in {(time.time()-t0)/60:.1f} min", flush=True)

    RAW.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP) as z:
        z.extractall(RAW)
    ZIP.unlink()                      # reclaim 7.5 GB immediately
    print("extracted; zip deleted. free disk:", f"{disk_free_gb():.1f} GB")

# The zip may nest everything one level deep -- descend to the real root.
kids = [d for d in RAW.iterdir() if d.is_dir()]
if len(kids) == 1 and not any(RAW.glob("*.xml")):
    RAW = kids[0]
print("RAW =", RAW)
print("folders:", sorted(d.name for d in RAW.iterdir() if d.is_dir())[:20])
print("xml files found:", sum(1 for _ in RAW.rglob("*.xml")))
"""),
    code("""
# Ultralytics brings its own torch; Kaggle images already have a CUDA torch.
!pip install -q ultralytics
import ultralytics, torch
print("ultralytics", ultralytics.__version__, "| torch", torch.__version__, "| cuda", torch.cuda.is_available())
"""),
    md("## 1. Audit — print the real label vocabulary before trusting any class map"),
    writefile("audit_dataset.py", HERE / "audit_dataset.py"),
    code("""
r = subprocess.run([sys.executable, "audit_dataset.py", str(RAW)], capture_output=True, text=True)
(WORK / "audit_report.txt").write_text(r.stdout + r.stderr)
print("\\n".join(r.stdout.split("\\n")[:90]))
print(f"\\n[full report -> {WORK/'audit_report.txt'}]")
"""),
    md("""
## 2. Convert VOC -> YOLO

Class comes from the XML `<object><name>`, never the folder name. Images are
downscaled to 1024 px (labels are normalized, so they stay valid). Near-duplicate
augmented copies are grouped by perceptual hash so they cannot straddle splits.

**A label string missing from the class map is a hard error** — a silently dropped
class is invisible accuracy loss.
"""),
    writefile("voc_to_yolo.py", HERE / "voc_to_yolo.py"),
    code("""
rc = subprocess.run([sys.executable, "voc_to_yolo.py", str(RAW), str(YOLO),
                     "--max-side", "1024", "--dup-threshold", "6", "--seed", "42"],
                    text=True).returncode
assert rc == 0, f"conversion failed rc={rc} (scroll up: likely an unmapped label)"
shutil.rmtree(RAW, ignore_errors=True)        # raw 7.5 GB no longer needed
print(open(YOLO / "data.yaml").read())
"""),
    md("## 3. Verify — range check plus a drawn overlay (catches x/y transposes)"),
    writefile("verify_labels.py", HERE / "verify_labels.py"),
    code("""
subprocess.run([sys.executable, "verify_labels.py", str(YOLO),
                "--sheet", str(WORK / "contact_sheet.jpg"), "-n", "16"], text=True)
from IPython.display import Image as IPyImage, display
display(IPyImage(str(WORK / "contact_sheet.jpg"), width=950))
"""),
    md("""
## 4. Train

Paper's hyperparameters: 640 px, 40 epochs, SGD, lr0 0.01, momentum 0.85,
warmup_momentum 0.7, mixup 0.2, effective batch 32.

`batch=16` + `nbs=32` gives the paper's effective batch of 32 through gradient
accumulation on **one** GPU — batch 32 outright would OOM a 16 GB T4, and
multi-GPU DDP is unreliable inside a notebook. Training runs from a .py file
for the same reason.

`yolov9c` (the paper's model) trains first and is saved before `yolov9s`
(small, for real-time CPU inference) starts, so a failure in the second
never costs you the first.
"""),
    writefile("train_yolo.py", HERE / "train_yolo.py"),
    code("""
rc = subprocess.run([sys.executable, "train_yolo.py", str(YOLO),
                     "--out", str(WORK), "--models", "yolov9c,yolov9s"], text=True).returncode
print("train_yolo.py exit code:", rc)
"""),
    md("## 5. Results"),
    code("""
import pandas as pd
for name in ("yolov9c", "yolov9s"):
    csv = WORK / "runs" / name / "results.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        df.columns = [c.strip() for c in df.columns]
        last = df.iloc[-1]
        print(f"{name}: mAP50={last.get('metrics/mAP50(B)'):.4f} "
              f"mAP50-95={last.get('metrics/mAP50-95(B)'):.4f} "
              f"P={last.get('metrics/precision(B)'):.4f} R={last.get('metrics/recall(B)'):.4f}")
print()
for f in sorted(WORK.glob('*.pt')) + sorted(WORK.glob('*.onnx')):
    print(f.name, round(f.stat().st_size/1e6, 1), "MB")
"""),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT} ({len(cells)} cells)")

# ---------------------------------------------------------------- second notebook
# 10-class path: data is already YOLO-format, so no audit/convert stage. Uses the
# Kaggle mirror of the same export the user has locally as archive.zip, so nothing
# needs uploading.
rf_cells = [
    md("""
# EEE_project -> YOLOv9 vehicle detection (10 classes)

Trains on the **already-converted** 10-class export — the same data as `archive.zip`,
mounted from Kaggle so there is nothing to upload.

Know what this is before you rely on it:

* **10 classes**, not 15 — no boat, horsecart, launch, tractor, wheelbarrow.
* **5,205 images**, not 9,058 (train 3,643 / valid 1,044 / test 518).
* Images were resized **640x640 "Stretch"**, which distorts aspect ratio. Run
  inference with `live_detect.py --stretch` so preprocessing matches training.

For the paper's 15-class result use `EEE_project_train_15class.ipynb` instead.
"""),
    code("""
import os, subprocess, sys, glob
from pathlib import Path

gpu = subprocess.run(["nvidia-smi","--query-gpu=name,memory.total","--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip()
print("GPU:", gpu or "NONE -- enable the T4 accelerator")

# Kaggle nests mounted datasets unpredictably (here: /kaggle/input/datasets/...),
# so search by STRUCTURE at any depth rather than assuming a level.
BASE = Path("/kaggle/input")
print("inputs:", sorted(str(p) for p in BASE.glob("*")))

RAW = None
for t in BASE.rglob("train"):                 # <root>/train/images
    if (t / "images").is_dir():
        RAW = t.parent; break
if RAW is None:
    for im in BASE.rglob("images"):           # <root>/images/train
        if (im / "train").is_dir():
            RAW = im.parent; break
if RAW is None:
    print("--- directories under /kaggle/input (depth <= 4) ---")
    for d in sorted(BASE.rglob("*")):
        rel = d.relative_to(BASE)
        if d.is_dir() and len(rel.parts) <= 4:
            print("   ", rel)
assert RAW is not None, "no YOLO-layout dataset found -- see the tree above"
WORK = Path("/kaggle/working"); WORK.mkdir(exist_ok=True)
print("RAW =", RAW)
for s_ in ("train", "valid", "val", "test"):
    d = RAW / s_ / "images"
    if d.is_dir():
        print(f"  {s_}: {sum(1 for _ in d.iterdir())} images")
"""),
    code("""
!pip install -q ultralytics
import ultralytics, torch
print("ultralytics", ultralytics.__version__, "| torch", torch.__version__, "| cuda", torch.cuda.is_available())
"""),
    writefile("train_yolo.py", HERE / "train_yolo.py"),
    code("""
rc = subprocess.run([sys.executable, "train_yolo.py", str(RAW),
                     "--out", str(WORK), "--models", "yolov9c,yolov9s"], text=True).returncode
print("train_yolo.py exit code:", rc)
"""),
    md("## Results"),
    code("""
import pandas as pd
for name in ("yolov9c", "yolov9s"):
    csv = WORK / "runs" / name / "results.csv"
    if csv.exists():
        df = pd.read_csv(csv); df.columns = [c.strip() for c in df.columns]
        last = df.iloc[-1]
        print(f"{name}: mAP50={last.get('metrics/mAP50(B)'):.4f} "
              f"mAP50-95={last.get('metrics/mAP50-95(B)'):.4f}")
for f in sorted(WORK.glob('*.pt')) + sorted(WORK.glob('*.onnx')) + sorted(WORK.glob('*classes.txt')):
    print(f.name, round(f.stat().st_size/1e6, 2), "MB")
"""),
]

rf_nb = {"cells": rf_cells, "metadata": nb["metadata"], "nbformat": 4, "nbformat_minor": 5}
RF_OUT = HERE.parent / "notebooks" / "EEE_project_train_10class.ipynb"
RF_OUT.write_text(json.dumps(rf_nb, indent=1))
print(f"wrote {RF_OUT} ({len(rf_cells)} cells)")
