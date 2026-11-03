"""Desktop window output."""
from __future__ import annotations

import time
from collections import Counter

import cv2

from .camera import Camera
from .detector import draw

WINDOW = "EEE_project - q to quit"


def run(camera: Camera, detector, classes, save_path=None) -> int:
    writer, last_seq, previous = None, -1, time.time()
    fps = 0.0
    try:
        while True:
            seq, frame = camera.read()
            if frame is None:
                break
            if seq == last_seq:
                time.sleep(0.005)
                continue
            dropped = seq - last_seq - 1 if last_seq >= 0 else 0
            last_seq = seq

            detections = detector(frame)
            draw(frame, detections, classes)

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, now - previous))
            previous = now

            counts = Counter(classes[d[4]] for d in detections)
            summary = "  ".join(f"{k}:{v}" for k, v in counts.most_common(6))
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 46), (0, 0, 0), -1)
            cv2.putText(frame, f"{fps:4.1f} FPS | {len(detections)} obj | dropped {dropped}",
                        (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, summary, (8, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA)

            if save_path:
                if writer is None:
                    writer = cv2.VideoWriter(str(save_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                             12, (frame.shape[1], frame.shape[0]))
                writer.write(frame)

            cv2.imshow(WINDOW, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
    return 0
