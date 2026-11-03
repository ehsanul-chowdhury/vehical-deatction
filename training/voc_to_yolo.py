#!/usr/bin/env python3
"""Convert the BD vehicle dataset (Pascal VOC XML) into a YOLO detection dataset.

Produces:
    <out>/data.yaml
    <out>/images/{train,val,test}/*.jpg
    <out>/labels/{train,val,test}/*.txt      "<cls> <xc> <yc> <w> <h>", normalized 0-1
    <out>/splits.json                        group -> split, for reproducibility

Five dataset-specific hazards this handles (all verified against real files):
  1. Class comes from XML <object><name> ("bike", "cng"), NOT the folder name
     ("Motorbike", "CNG"). A folder-based converter mislabels every box.
  2. Images are 12 MP phone photos -> downscaled (labels are normalized, so
     resizing an image does not change its label file).
  3. Filenames contain spaces and parentheses: "mixed (1).jpg" -> "mixed_1.jpg".
  4. Boxes are 1-based and xmax can equal full width -> clamped, degenerates dropped.
  5. 1,791 images are augmented copies of others -> grouped by perceptual hash so
     a photo and its flipped/recoloured copies never straddle a split boundary.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from audit_dataset import find_pairs

# Paper's class order -- keep it so per-class mAP is directly comparable.
CLASSES = [
    "bicycle", "bike", "boat", "bus", "car", "cng", "easybike", "horsecart",
    "launch", "leguna", "rickshaw", "tractor", "truck", "van", "wheelbarrow",
]
CLASS_ID = {c: i for i, c in enumerate(CLASSES)}

# Applied AFTER squashing case and separators, so "Easy-bike"/"easy bike" -> "easybike".
SYNONYMS = {
    "motorbike": "bike", "motorcycle": "bike", "motor": "bike", "bikes": "bike",
    "laguna": "leguna", "leguna": "leguna", "lagoona": "leguna",
    "lunch": "launch", "launches": "launch",
    "nouka": "boat", "boats": "boat", "nao": "boat",
    "cars": "car", "buss": "bus", "buses": "bus", "trucks": "truck",
    "cycle": "bicycle", "bycycle": "bicycle", "bicyle": "bicycle",
    "horsecarts": "horsecart", "horse": "horsecart",
    "easybikes": "easybike", "wheelbarrows": "wheelbarrow",
    "threewheeler": "cng", "autorickshaw": "cng",
}


def canon_label(raw: str) -> str:
    """'Easy-bike ' -> 'easybike'; 'Motorbike' -> 'bike'."""
    s = re.sub(r"[^a-z0-9]", "", (raw or "").strip().lower())
    return SYNONYMS.get(s, s)


def sanitize(folder: str, stem: str) -> str:
    """'Multi-class Vehicles' + 'mixed (1)' -> 'multiclassvehicles__mixed_1'."""
    f = re.sub(r"[^A-Za-z0-9]+", "", folder).lower()
    s = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_").lower()
    return f"{f}__{s}" if f else s


def dhash(im: Image.Image, size: int = 8) -> int:
    """64-bit difference hash. Mirror-canonicalized by the caller."""
    g = im.convert("L").resize((size + 1, size), Image.LANCZOS)
    px = list(g.getdata())
    bits = 0
    for r in range(size):
        row = px[r * (size + 1):(r + 1) * (size + 1)]
        for c in range(size):
            bits = (bits << 1) | (1 if row[c] < row[c + 1] else 0)
    return bits


def canon_hash(im: Image.Image) -> int:
    """min(hash, hash-of-mirror) so a horizontal flip collides with its original."""
    return min(dhash(im), dhash(im.transpose(Image.FLIP_LEFT_RIGHT)))


# ---------------------------------------------------------------- worker

_STAGE: dict = {}


def _init(stage_img: str, stage_lbl: str, max_side: int, quality: int, keep_difficult: bool):
    _STAGE.update(img=Path(stage_img), lbl=Path(stage_lbl), max_side=max_side,
                  quality=quality, keep_difficult=keep_difficult)


def convert_one(task: tuple[str, str, str]):
    """Parse one XML + image -> staged jpg + txt. Returns a record or an error."""
    img_path, xml_path, folder = Path(task[0]), Path(task[1]), task[2]
    try:
        objs = ET.parse(xml_path).getroot().findall("object")
    except Exception as exc:
        return {"err": f"{xml_path.name}: unparseable XML ({exc})"}

    try:
        im = Image.open(img_path)
        im.load()
    except Exception as exc:
        return {"err": f"{img_path.name}: unreadable image ({exc})"}

    if im.mode != "RGB":
        im = im.convert("RGB")
    W, H = im.size  # TRUE dims -- never trust XML <size>
    if W < 2 or H < 2:
        return {"err": f"{img_path.name}: degenerate image {W}x{H}"}

    lines, classes, unknown = [], [], []
    for obj in objs:
        if not _STAGE["keep_difficult"] and (obj.findtext("difficult") or "0").strip() == "1":
            continue
        label = canon_label(obj.findtext("name"))
        if label not in CLASS_ID:
            unknown.append(obj.findtext("name"))
            continue
        bb = obj.find("bndbox")
        if bb is None:
            continue
        try:
            x1 = float(bb.findtext("xmin")); y1 = float(bb.findtext("ymin"))
            x2 = float(bb.findtext("xmax")); y2 = float(bb.findtext("ymax"))
        except (TypeError, ValueError):
            continue
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        x1 = max(0.0, min(x1, W)); x2 = max(0.0, min(x2, W))
        y1 = max(0.0, min(y1, H)); y2 = max(0.0, min(y2, H))
        if (x2 - x1) < 2 or (y2 - y1) < 2:   # 1-based coords + clamping make this real
            continue
        xc, yc = ((x1 + x2) / 2) / W, ((y1 + y2) / 2) / H
        bw, bh = (x2 - x1) / W, (y2 - y1) / H
        if not (0 < bw <= 1 and 0 < bh <= 1 and 0 <= xc <= 1 and 0 <= yc <= 1):
            continue
        lines.append(f"{CLASS_ID[label]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        classes.append(label)

    if unknown:
        return {"err": f"{xml_path.name}: UNKNOWN LABELS {sorted(set(unknown))}"}
    if not lines:
        return {"err": f"{img_path.name}: no usable boxes", "soft": True}

    h = canon_hash(im)

    max_side = _STAGE["max_side"]
    if max(W, H) > max_side:
        scale = max_side / max(W, H)
        im = im.resize((max(1, round(W * scale)), max(1, round(H * scale))), Image.LANCZOS)

    stem = sanitize(folder, img_path.stem)
    im.save(_STAGE["img"] / f"{stem}.jpg", "JPEG", quality=_STAGE["quality"], optimize=True)
    (_STAGE["lbl"] / f"{stem}.txt").write_text("\n".join(lines) + "\n")
    im.close()
    return {"stem": stem, "hash": h, "classes": classes}


# ---------------------------------------------------------------- grouping

def cluster(hashes: list[int], threshold: int) -> list[int]:
    """Union-find over perceptual hashes within `threshold` Hamming bits.

    threshold=0 groups only byte-identical hashes (exact dupes + flips).
    """
    n = len(hashes)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    # exact-hash collisions first (cheap, catches dupes and mirrored copies)
    by_hash: dict[int, list[int]] = collections.defaultdict(list)
    for i, h in enumerate(hashes):
        by_hash[h].append(i)
    for idxs in by_hash.values():
        for j in idxs[1:]:
            union(idxs[0], j)

    if threshold > 0:
        uniq = sorted(by_hash)
        rep = {h: by_hash[h][0] for h in uniq}
        arr = np.array(uniq, dtype=">u8").view(np.uint8).reshape(-1, 8)
        popc = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1).astype(np.uint8)
        step = 512
        for s in range(0, len(arr), step):
            block = arr[s:s + step]
            dist = popc[block[:, None, :] ^ arr[None, :, :]].sum(-1)
            for bi, gj in zip(*np.where(dist <= threshold)):
                i, j = s + int(bi), int(gj)
                if i < j:
                    union(rep[uniq[i]], rep[uniq[j]])

    return [find(i) for i in range(n)]


def split_groups(records, group_of, ratios, seed):
    """Group-level split, stratified on each group's RAREST class.

    Stratifying on the rarest class is what keeps Leguna (218), Wheelbarrow (237)
    and Horse-cart (256) present in val and test.
    """
    freq: collections.Counter = collections.Counter()
    for r in records:
        freq.update(set(r["classes"]))

    members: dict[int, list] = collections.defaultdict(list)
    for r, g in zip(records, group_of):
        members[g].append(r)

    buckets: dict[str, list[int]] = collections.defaultdict(list)
    for g, rs in members.items():
        present = {c for r in rs for c in r["classes"]}
        buckets[min(present, key=lambda c: freq[c])].append(g)

    rng = random.Random(seed)
    assign: dict[int, str] = {}
    tr, va = ratios
    for cls in sorted(buckets):
        gs = sorted(buckets[cls])
        rng.shuffle(gs)
        n = len(gs)
        n_tr = max(1, round(n * tr)) if n > 2 else n
        n_va = max(1, round(n * va)) if n > 2 else 0
        if n > 2:
            n_tr = min(n_tr, n - 2)
            n_va = min(n_va, n - n_tr - 1)
        for i, g in enumerate(gs):
            assign[g] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
    return assign


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--quality", type=int, default=90)
    ap.add_argument("--dup-threshold", type=int, default=6,
                    help="Hamming bits for near-duplicate grouping; 0 = exact+flip only")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)))
    ap.add_argument("--keep-difficult", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="debug: convert only N images")
    args = ap.parse_args()

    paired, img_only, xml_only = find_pairs(args.root)
    if not paired:
        print(f"FATAL: no image+xml pairs under {args.root}", file=sys.stderr)
        return 1
    if args.limit:
        paired = paired[: args.limit]
    print(f"pairs={len(paired)}  image-only={len(img_only)}  xml-only={len(xml_only)}")

    stage_img = args.out / "_stage_images"
    stage_lbl = args.out / "_stage_labels"
    for d in (stage_img, stage_lbl):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)

    tasks = []
    for img, xml in paired:
        try:
            folder = img.relative_to(args.root).parts[0] if len(img.relative_to(args.root).parts) > 1 else ""
        except ValueError:
            folder = img.parent.name
        tasks.append((str(img), str(xml), folder))

    records, errors, soft = [], [], []
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=_init,
        initargs=(str(stage_img), str(stage_lbl), args.max_side, args.quality, args.keep_difficult),
    ) as ex:
        for i, res in enumerate(ex.map(convert_one, tasks, chunksize=16), 1):
            if "err" in res:
                (soft if res.get("soft") else errors).append(res["err"])
            else:
                records.append(res)
            if i % 500 == 0:
                print(f"  converted {i}/{len(tasks)}", flush=True)

    print(f"\nconverted={len(records)}  skipped_no_boxes={len(soft)}  errors={len(errors)}")
    for e in errors[:20]:
        print(f"  ERR {e}")

    # An unmapped label is a silently-lost class, so refuse to continue.
    unknown = [e for e in errors if "UNKNOWN LABELS" in e]
    if unknown:
        print(f"\nFATAL: {len(unknown)} file(s) carry labels missing from the class map.", file=sys.stderr)
        print("Add them to SYNONYMS (or CLASSES) and re-run. First few:", file=sys.stderr)
        for e in unknown[:10]:
            print(f"  {e}", file=sys.stderr)
        return 2
    if not records:
        print("FATAL: nothing converted", file=sys.stderr)
        return 1

    group_of = cluster([r["hash"] for r in records], args.dup_threshold)
    n_groups = len(set(group_of))
    print(f"grouping: {len(records)} images -> {n_groups} groups "
          f"({len(records) - n_groups} near-duplicates absorbed)")

    assign = split_groups(records, group_of, (0.8, 0.1), args.seed)

    for sub in ("images", "labels"):
        for sp in ("train", "val", "test"):
            (args.out / sub / sp).mkdir(parents=True, exist_ok=True)

    per_split: collections.Counter = collections.Counter()
    per_split_cls: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    splits_json = {}
    for r, g in zip(records, group_of):
        sp = assign[g]
        stem = r["stem"]
        shutil.move(str(stage_img / f"{stem}.jpg"), str(args.out / "images" / sp / f"{stem}.jpg"))
        shutil.move(str(stage_lbl / f"{stem}.txt"), str(args.out / "labels" / sp / f"{stem}.txt"))
        per_split[sp] += 1
        per_split_cls[sp].update(r["classes"])
        splits_json[stem] = {"split": sp, "group": int(g)}

    shutil.rmtree(stage_img, ignore_errors=True)
    shutil.rmtree(stage_lbl, ignore_errors=True)

    (args.out / "data.yaml").write_text(
        f"path: {args.out.resolve()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        f"nc: {len(CLASSES)}\nnames: [{', '.join(CLASSES)}]\n"
    )
    (args.out / "splits.json").write_text(json.dumps(splits_json, indent=1))

    print("\n--- images per split ---")
    for sp in ("train", "val", "test"):
        print(f"  {sp:5s} {per_split[sp]:6d}")
    print("\n--- boxes per class per split ---")
    print(f"  {'class':<12}{'train':>8}{'val':>7}{'test':>7}")
    missing = []
    for c in CLASSES:
        t, v, s = per_split_cls['train'][c], per_split_cls['val'][c], per_split_cls['test'][c]
        print(f"  {c:<12}{t:>8}{v:>7}{s:>7}")
        if v == 0 or s == 0:
            missing.append(c)
    if missing:
        print(f"\n  WARNING: absent from val or test: {missing}")
    print(f"\nwrote {args.out / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
