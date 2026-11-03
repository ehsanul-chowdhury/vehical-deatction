#!/usr/bin/env python3
"""Read-only audit of the BD vehicle Pascal VOC dataset.

Run this BEFORE converting. Its most important output is a histogram of every
raw <object><name> string in the corpus: the class map in voc_to_yolo.py must be
built against that reality, not against the Mendeley folder names (which read
"Motorbike"/"CNG"/"Easy-bike" while the XML actually says "bike"/"cng"/"easybike").

Writes nothing. Exits 0 always -- this is a report, not a gate.
"""
from __future__ import annotations

import argparse
import collections
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    from PIL import Image
except ImportError:  # pillow is present on Kaggle; keep the audit usable without it
    Image = None

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def find_pairs(root: Path):
    """Index images and annotations by (directory, stem), pairing within a directory.

    Pairing per-directory rather than globally matters because stems repeat across
    class folders in this dataset (every folder has its own "(1)", "(2)", ...).
    """
    imgs: dict[tuple[str, str], Path] = {}
    xmls: dict[tuple[str, str], Path] = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            p = Path(dirpath) / fn
            ext = p.suffix.lower()
            if ext in IMG_EXT:
                imgs[(dirpath, p.stem)] = p
            elif ext == ".xml":
                xmls[(dirpath, p.stem)] = p
    both = imgs.keys() & xmls.keys()
    return (
        sorted((imgs[k], xmls[k]) for k in both),
        sorted(imgs[k] for k in imgs.keys() - both),
        sorted(xmls[k] for k in xmls.keys() - both),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="dataset root to walk")
    args = ap.parse_args()

    if not args.root.exists():
        print(f"FATAL: {args.root} does not exist", file=sys.stderr)
        return 1

    paired, img_only, xml_only = find_pairs(args.root)
    print("=" * 70)
    print(f"ROOT: {args.root}")
    print(f"paired image+xml : {len(paired)}")
    print(f"image without xml: {len(img_only)}")
    print(f"xml without image: {len(xml_only)}")

    print("\n--- top-level directories (image counts) ---")
    per_dir: collections.Counter = collections.Counter()
    for img, _ in paired:
        try:
            rel = img.relative_to(args.root).parts
        except ValueError:
            rel = img.parts
        per_dir[rel[0] if len(rel) > 1 else "<root>"] += 1
    for d, n in sorted(per_dir.items(), key=lambda kv: -kv[1]):
        print(f"  {n:6d}  {d}")

    labels: collections.Counter = collections.Counter()
    boxes_per_img: collections.Counter = collections.Counter()
    dims: collections.Counter = collections.Counter()
    difficult = truncated = 0
    size_mismatch: list[str] = []
    bad_box: list[str] = []
    unparseable: list[str] = []
    unreadable_img: list[str] = []

    for img, xml in paired:
        try:
            root_el = ET.parse(xml).getroot()
        except Exception as exc:
            unparseable.append(f"{xml}: {exc}")
            continue

        # XML-declared size
        xw = xh = None
        size_el = root_el.find("size")
        if size_el is not None:
            try:
                xw = int(float(size_el.findtext("width", "0")))
                xh = int(float(size_el.findtext("height", "0")))
            except (TypeError, ValueError):
                xw = xh = None

        # true size on disk -- the converter must trust this one, not the XML
        tw = th = None
        if Image is not None:
            try:
                with Image.open(img) as im:
                    tw, th = im.size
            except Exception as exc:
                unreadable_img.append(f"{img}: {exc}")
        if tw and xw and (tw, th) != (xw, xh):
            size_mismatch.append(f"{img.name}: xml={xw}x{xh} actual={tw}x{th}")
        if tw:
            dims[f"{tw}x{th}"] += 1

        objs = root_el.findall("object")
        boxes_per_img[len(objs)] += 1
        W, H = (tw or xw or 0), (th or xh or 0)
        for obj in objs:
            labels[(obj.findtext("name") or "<MISSING>").strip()] += 1
            if (obj.findtext("difficult") or "0").strip() == "1":
                difficult += 1
            if (obj.findtext("truncated") or "0").strip() == "1":
                truncated += 1
            bb = obj.find("bndbox")
            if bb is None:
                bad_box.append(f"{img.name}: object with no bndbox")
                continue
            try:
                x1 = float(bb.findtext("xmin")); y1 = float(bb.findtext("ymin"))
                x2 = float(bb.findtext("xmax")); y2 = float(bb.findtext("ymax"))
            except (TypeError, ValueError) as exc:
                bad_box.append(f"{img.name}: unparseable bndbox ({exc})")
                continue
            if x2 <= x1 or y2 <= y1:
                bad_box.append(f"{img.name}: degenerate {x1},{y1},{x2},{y2}")
            elif W and H and (x1 < 0 or y1 < 0 or x2 > W or y2 > H):
                bad_box.append(f"{img.name}: out of bounds {x1},{y1},{x2},{y2} in {W}x{H}")

    print("\n" + "=" * 70)
    print("RAW <object><name> HISTOGRAM  <-- build the class map from THIS")
    print("=" * 70)
    for name, n in sorted(labels.items(), key=lambda kv: -kv[1]):
        print(f"  {n:7d}  {name!r}")
    print(f"\n  distinct label strings: {len(labels)}")
    print(f"  total boxes           : {sum(labels.values())}")

    print("\n--- boxes per image ---")
    for k in sorted(boxes_per_img):
        print(f"  {boxes_per_img[k]:6d} images have {k} box(es)")

    print("\n--- 10 most common image dimensions ---")
    for d, n in dims.most_common(10):
        print(f"  {n:6d}  {d}")

    print(f"\n--- flags ---  difficult={difficult}  truncated={truncated}")
    for title, items in (
        ("XML size != actual size", size_mismatch),
        ("bad boxes", bad_box),
        ("unparseable XML", unparseable),
        ("unreadable images", unreadable_img),
        ("image without xml", [str(p) for p in img_only]),
        ("xml without image", [str(p) for p in xml_only]),
    ):
        print(f"\n--- {title}: {len(items)} ---")
        for line in items[:15]:
            print(f"    {line}")
        if len(items) > 15:
            print(f"    ... and {len(items) - 15} more")
    return 0


if __name__ == "__main__":
    sys.exit(main())
