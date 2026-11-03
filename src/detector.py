"""YOLOv9 ONNX inference."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

FALLBACK_CLASSES = [
    "bicycle", "bike", "boat", "bus", "car", "cng", "easybike", "horsecart",
    "launch", "leguna", "rickshaw", "tractor", "truck", "van", "wheelbarrow",
]


def load_classes(model: Path, explicit: Path | None = None) -> list[str]:
    """Read class names from an explicit file, a sibling .classes.txt, or a data.yaml.

    The 10- and 15-class models order their classes differently, so guessing
    here would mislabel every detection.
    """
    candidates = [
        explicit,
        model.parent / f"{model.stem}.classes.txt",
        model.parent / "classes.txt",
        model.parent / "data.yaml",
    ]
    for cand in candidates:
        if cand is None or not Path(cand).exists():
            continue
        cand = Path(cand)
        text = cand.read_text()

        if cand.suffix in (".yaml", ".yml"):
            for line in text.splitlines():
                line = line.strip()
                if line.startswith("names:") and "[" in line:
                    inner = line.split("[", 1)[1].rsplit("]", 1)[0]
                    names = [x.strip().strip("'\"") for x in inner.split(",") if x.strip()]
                    if names:
                        return names
        else:
            names = [l.strip() for l in text.splitlines() if l.strip()]
            if names:
                return names

    return list(FALLBACK_CLASSES)


def color_for(index: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(index * 97 + 11)
    return tuple(int(v) for v in rng.integers(60, 255, 3))


def letterbox(img, size: int):
    """Resize preserving aspect ratio and pad to square."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = round(h * r), round(w * r)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = cv2.resize(
        img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return canvas, r, left, top


def draw(frame, detections, classes) -> None:
    """Draw boxes and labels onto the frame in place."""
    for x1, y1, x2, y2, cid, score in detections:
        color = color_for(cid)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{classes[cid]} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, max(10, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


class Detector:
    """Runs an exported YOLOv9 ONNX model.

    Uses onnxruntime when available and falls back to OpenCV's DNN module,
    which keeps PyTorch out of the local install.
    """

    def __init__(self, model: Path, imgsz: int = 640, conf: float = 0.35,
                 iou: float = 0.45, classes: list[str] | None = None,
                 stretch: bool = False, providers: list[str] | None = None):
        self.imgsz, self.conf, self.iou = imgsz, conf, iou
        self.classes = list(classes) if classes else list(FALLBACK_CLASSES)
        self.stretch = stretch
        self.session = None

        try:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 0
            available = ort.get_available_providers()
            wanted = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
            use = [p for p in wanted if p in available] or ["CPUExecutionProvider"]
            self.session = ort.InferenceSession(str(model), opts, providers=use)
            self.input_name = self.session.get_inputs()[0].name
            self.backend = f"onnxruntime:{use[0].replace('ExecutionProvider', '')}"
        except Exception:
            self.net = cv2.dnn.readNetFromONNX(str(model))
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self.backend = "opencv-dnn:CPU"

    def _forward(self, blob: np.ndarray) -> np.ndarray:
        if self.session is not None:
            return self.session.run(None, {self.input_name: blob})[0]
        self.net.setInput(blob)
        return self.net.forward()

    def __call__(self, frame):
        nc = len(self.classes)

        if self.stretch:
            # The 10-class export was resized "640x640 (Stretch)". Letterboxing
            # here instead would be a train/serve preprocessing mismatch.
            h, w = frame.shape[:2]
            img = cv2.resize(frame, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
            rx, ry, padx, pady = self.imgsz / w, self.imgsz / h, 0.0, 0.0
        else:
            img, r, padx, pady = letterbox(frame, self.imgsz)
            rx = ry = r

        blob = img[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = np.asarray(self._forward(np.ascontiguousarray(blob)))
        if out.ndim == 3:
            out = out[0]

        # Ultralytics exports (4+nc, N). Decide orientation by shape, not by nc:
        # the channel axis is always far smaller than the anchor axis.
        if out.shape[0] < out.shape[1]:
            out = out.T
        if out.shape[1] != 4 + nc:
            raise RuntimeError(
                f"ONNX output {out.shape} implies {out.shape[1] - 4} classes but the "
                f"class list has {nc}. Point --names at the matching .classes.txt.")

        scores = out[:, 4:]
        cls_id = scores.argmax(1)
        cls_score = scores[np.arange(len(scores)), cls_id]
        keep = cls_score >= self.conf
        if not keep.any():
            return []

        box, cls_id, cls_score = out[keep, :4], cls_id[keep], cls_score[keep]

        # cx,cy,w,h in model pixels -> x,y,w,h in original frame pixels
        x = (box[:, 0] - box[:, 2] / 2 - padx) / rx
        y = (box[:, 1] - box[:, 3] / 2 - pady) / ry
        w, h = box[:, 2] / rx, box[:, 3] / ry
        rects = np.stack([x, y, w, h], 1)

        idxs = cv2.dnn.NMSBoxes(rects.tolist(), cls_score.tolist(), self.conf, self.iou)
        if len(idxs) == 0:
            return []

        H, W = frame.shape[:2]
        results = []
        for i in np.asarray(idxs).flatten():
            bx, by, bw, bh = rects[i]
            x1, y1 = max(0, int(bx)), max(0, int(by))
            x2, y2 = min(W - 1, int(bx + bw)), min(H - 1, int(by + bh))
            if x2 > x1 and y2 > y1:
                results.append((x1, y1, x2, y2, int(cls_id[i]), float(cls_score[i])))
        return results
