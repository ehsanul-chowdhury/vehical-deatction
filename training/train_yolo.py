#!/usr/bin/env python3
"""Fine-tune YOLOv9 on a YOLO-format dataset and export ONNX.

Works with either dataset in this project:
  * the raw VOC conversion produced by voc_to_yolo.py (15 classes), or
  * a pre-converted export such as archive.zip (10 classes, train/valid/test).

Two things this fixes that bite people with pre-converted exports:

  1. Roboflow writes `train: ../train/images` into a data.yaml that sits at the
     dataset ROOT. Ultralytics resolves that relative to the yaml, so `../`
     escapes the dataset and training dies with "no images found". We always
     rewrite data.yaml with absolute paths.
  2. Roboflow names the validation folder `valid`; Ultralytics expects `val`.
     Both are detected.

Paper hyperparameters (arXiv:2410.08230): 640 px, 40 epochs, SGD, lr0 0.01,
momentum 0.85, warmup_momentum 0.7, mixup 0.2, effective batch 32.
`batch=16` + `nbs=32` reaches that effective batch by gradient accumulation --
batch 32 outright OOMs a 16 GB T4.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def find_split(root: Path, *names: str) -> Path | None:
    """Locate a split's image dir under either common YOLO layout.

    Roboflow writes <root>/train/images; voc_to_yolo.py (and the Ultralytics
    docs) write <root>/images/train. Both are valid, so check both.
    """
    for n in names:
        for d in (root / n / "images", root / "images" / n):
            if d.is_dir() and any(d.iterdir()):
                return d
    return None


def read_names(root: Path) -> list[str] | None:
    """Pull class names out of an existing data.yaml without needing a yaml parser."""
    for cand in (root / "data.yaml", root / "data.yml"):
        if not cand.exists():
            continue
        for line in cand.read_text().splitlines():
            line = line.strip()
            if line.startswith("names:") and "[" in line:
                inner = line.split("[", 1)[1].rsplit("]", 1)[0]
                names = [p.strip().strip("'\"") for p in inner.split(",") if p.strip()]
                if names:
                    return names
    return None


def build_yaml(root: Path, out: Path) -> tuple[Path, list[str]]:
    train = find_split(root, "train")
    val = find_split(root, "val", "valid")
    test = find_split(root, "test")
    if train is None or val is None:
        sys.exit(f"FATAL: need train/images and val|valid/images under {root}")

    names = read_names(root)
    if names is None:
        sys.exit(f"FATAL: no class names found in {root}/data.yaml -- cannot guess them")

    lines = [
        f"path: {root.resolve()}",
        f"train: {train.resolve()}",
        f"val: {val.resolve()}",
    ]
    if test is not None:
        lines.append(f"test: {test.resolve()}")
    lines += [f"nc: {len(names)}", "names: [" + ", ".join(names) + "]"]
    out.write_text("\n".join(lines) + "\n")
    return out, names


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="dataset root (contains train/ val|valid/ test/)")
    ap.add_argument("--models", default="yolov9c,yolov9s",
                    help="comma list; trained in order, each saved before the next starts")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--nbs", type=int, default=32, help="nominal batch -> gradient accumulation")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", type=Path, default=Path("/kaggle/working"))
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    from ultralytics import YOLO

    args.out.mkdir(parents=True, exist_ok=True)
    data_yaml, names = build_yaml(args.root, args.out / "data_abs.yaml")
    print(f"dataset : {args.root}")
    print(f"classes : {len(names)} -> {names}")
    print(f"data.yaml rewritten with absolute paths -> {data_yaml}\n")

    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"\n{'=' * 60}\nTRAINING {name}\n{'=' * 60}", flush=True)
        try:
            model = YOLO(f"{name}.pt")
            model.train(
                data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, nbs=args.nbs,
                optimizer="SGD", lr0=0.01, lrf=0.01,
                momentum=0.85, warmup_momentum=0.7, mixup=0.2,
                cache=False, workers=args.workers, patience=15, seed=42,
                device=args.device,
                project=str(args.out / "runs"), name=name, exist_ok=True,
            )
            best = args.out / "runs" / name / "weights" / "best.pt"
            if not best.exists():
                print(f"!! {name}: no best.pt produced", flush=True)
                continue
            shutil.copy(best, args.out / f"{name}_best.pt")

            m = YOLO(str(best))
            if "test:" in data_yaml.read_text():
                m.val(data=str(data_yaml), split="test",
                      project=str(args.out / "runs"), name=f"{name}_test", exist_ok=True)
            onnx = m.export(format="onnx", imgsz=args.imgsz, opset=12, simplify=True)
            shutil.copy(onnx, args.out / f"{name}_best.onnx")
            # Inference must know the exact class order this model was trained on;
            # the 10-class cut and the 15-class build differ.
            (args.out / f"{name}_best.classes.txt").write_text("\n".join(names) + "\n")
            print(f"SAVED {name}: {name}_best.pt + {name}_best.onnx", flush=True)
        except Exception as exc:
            print(f"!! {name} FAILED: {type(exc).__name__}: {exc}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
