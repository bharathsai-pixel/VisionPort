"""Video playback, frame extraction, and frame-by-frame detection."""

from collections import Counter
from pathlib import Path

import cv2
import numpy as np

import traffic_analytics
import vehicle_detection


def play_video(video_path: str | Path) -> bool:
    """Play a video at its recorded rate; press Q to exit."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        return False
    fps = capture.get(cv2.CAP_PROP_FPS)
    delay = max(1, round(1000 / fps)) if fps and fps > 0 else 33
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            cv2.imshow("VisionPort - Video (Q to quit)", frame)
            if cv2.waitKey(delay) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return True


def extract_frames(video_path: str | Path, output_dir: str | Path) -> int:
    """Save every readable video frame as a numbered JPG; return saved frame count."""
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        return 0
    saved = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            next_number = saved + 1
            if cv2.imwrite(str(folder / f"frame_{next_number:06d}.jpg"), frame):
                saved = next_number
    finally:
        capture.release()
    return saved


def process_video(video_path: str | Path, output_path: str | Path) -> dict:
    """Run YOLO on every video frame and save an annotated MP4 and statistics."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError("OpenCV could not read this video. Try an MP4 or AVI file.")
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        raise ValueError("The uploaded video has invalid dimensions.")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("OpenCV could not create the output video.")

    category_totals = Counter()
    frame_count = 0
    frame_vehicle_total = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            annotated, detections = vehicle_detection.detect_vehicles(frame)
            stats = traffic_analytics.frame_statistics(detections)
            cv2.putText(annotated, f"Vehicles: {stats['total']}  Density: {stats['density']}",
                        (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (70, 240, 190), 2,
                        cv2.LINE_AA)
            writer.write(annotated)
            frame_count += 1
            frame_vehicle_total += stats["total"]
            category_totals.update(stats["counts"])
    except Exception:
        capture.release()
        writer.release()
        destination.unlink(missing_ok=True)
        raise
    finally:
        capture.release()
        writer.release()

    if frame_count == 0:
        destination.unlink(missing_ok=True)
        raise ValueError("The uploaded video contains no readable frames.")
    stats = traffic_analytics.video_statistics(category_totals, frame_count, frame_vehicle_total)
    stats.update({"fps": round(float(fps), 2), "width": width, "height": height,
                  "output_path": destination})
    return stats
