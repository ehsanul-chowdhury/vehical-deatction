#!/usr/bin/env python3
"""Verify a converted YOLO dataset: numeric sanity + a visual contact sheet.

The range check catches malformed values; only the drawn overlay catches an
x/y transpose or a corner-vs-centre mistake, which are otherwise invisible.
"""
from __future__ import annotations

import argparse, random, sys
from pathlib import Path
from PIL import Image, ImageDraw

CLASSES = ["bicycle","bike","boat","bus","car","cng","easybike","horsecart",
           "launch","leguna","rickshaw","tractor","truck","van","wheelbarrow"]
PALETTE = ["#e6194b","#3cb44b","#ffe119","#4363d8","#f58231","#911eb4","#46f0f0",
           "#f032e6","#bcf60c","#fabebe","#008080","#e6beff","#9a6324","#fffac8","#800000"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--sheet", type=Path, default=Path("contact_sheet.jpg"))
    ap.add_argument("-n", type=int, default=12)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--cell", type=int, default=420)
    args = ap.parse_args()

    problems, total_boxes, picks = [], 0, []
    for split in ("train", "val", "test"):
        idir, ldir = args.root / "images" / split, args.root / "labels" / split
        if not idir.exists():
            continue
        images = sorted(idir.glob("*.jpg"))
        labels = sorted(ldir.glob("*.txt"))
        istems, lstems = {p.stem for p in images}, {p.stem for p in labels}
        if istems != lstems:
            problems.append(f"{split}: {len(istems-lstems)} images w/o label, "
                            f"{len(lstems-istems)} labels w/o image")
        for lp in labels:
            for ln, line in enumerate(lp.read_text().split("\n"), 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    problems.append(f"{lp.name}:{ln} has {len(parts)} fields, want 5")
                    continue
                try:
                    cid = int(parts[0]); vals = [float(v) for v in parts[1:]]
                except ValueError:
                    problems.append(f"{lp.name}:{ln} non-numeric: {line!r}")
                    continue
                total_boxes += 1
                if not 0 <= cid < len(CLASSES):
                    problems.append(f"{lp.name}:{ln} class_id {cid} out of range")
                if any(not (0.0 <= v <= 1.0) for v in vals):
                    problems.append(f"{lp.name}:{ln} value outside [0,1]: {vals}")
                xc, yc, w, h = vals
                if w <= 0 or h <= 0:
                    problems.append(f"{lp.name}:{ln} non-positive w/h: {w},{h}")
                if xc - w/2 < -1e-6 or xc + w/2 > 1+1e-6 or yc - h/2 < -1e-6 or yc + h/2 > 1+1e-6:
                    problems.append(f"{lp.name}:{ln} box escapes frame: {vals}")
        picks += [(p, ldir / f"{p.stem}.txt") for p in images]

    print(f"total boxes checked: {total_boxes}")
    print(f"problems: {len(problems)}")
    for p in problems[:25]:
        print(f"  {p}")

    random.seed(0)
    random.shuffle(picks)
    picks = picks[: args.n]
    if picks:
        cols = min(args.cols, len(picks))
        rows = (len(picks) + cols - 1) // cols
        cell = args.cell
        sheet = Image.new("RGB", (cols * cell, rows * cell), "#111111")
        for i, (ip, lp) in enumerate(picks):
            with Image.open(ip) as im:
                im = im.convert("RGB")
                W, H = im.size
                d = ImageDraw.Draw(im)
                if lp.exists():
                    for line in lp.read_text().split("\n"):
                        if not line.strip():
                            continue
                        cid, xc, yc, w, h = line.split()
                        cid = int(cid); xc, yc, w, h = map(float, (xc, yc, w, h))
                        x1, y1 = (xc - w/2) * W, (yc - h/2) * H
                        x2, y2 = (xc + w/2) * W, (yc + h/2) * H
                        col = PALETTE[cid % len(PALETTE)]
                        d.rectangle([x1, y1, x2, y2], outline=col, width=max(2, W // 180))
                        d.text((x1 + 4, max(0, y1 + 2)), CLASSES[cid], fill=col)
                im.thumbnail((cell, cell), Image.LANCZOS)
                sheet.paste(im, (i % cols * cell + (cell - im.width) // 2,
                                 i // cols * cell + (cell - im.height) // 2))
        sheet.save(args.sheet, "JPEG", quality=88)
        print(f"\ncontact sheet -> {args.sheet}  ({len(picks)} images)")
    return 1 if problems else 0

if __name__ == "__main__":
    sys.exit(main())
