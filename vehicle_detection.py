"""Reusable YOLOv8 vehicle inference for images and video frames."""

import threading
from pathlib import Path

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "yolov8n.pt"
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
_model = None
_model_lock = threading.Lock()
_tracker_lock = threading.Lock()
_tracks = {}
_next_track_id = 1
_tracker_frame = 0
_TRACK_MAX_AGE = 30
_TRACK_IOU_THRESHOLD = 0.2


def get_model():
    """Load YOLOv8n once and reuse it for subsequent inference calls."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                try:
                    from ultralytics import YOLO
                    _model = YOLO(str(MODEL_PATH) if MODEL_PATH.is_file() else "yolov8n.pt")
                except Exception as exc:
                    if MODEL_PATH.is_file():
                        guidance = (
                            "The local yolov8n.pt weights were found, but the YOLO/PyTorch "
                            "runtime could not be imported. Repair the project environment "
                            "from an activated venv with `python -m pip install --force-reinstall "
                            "-r requirements.txt`. If pip itself is broken, recreate the venv "
                            "first (remove venv, then run `py -m venv venv` and install the "
                            "requirements again)."
                        )
                    else:
                        guidance = (
                            "Could not load YOLOv8n. Connect to the internet for its first "
                            "download or place yolov8n.pt in the project folder, and check "
                            "that the PyTorch runtime is installed correctly."
                        )
                    raise RuntimeError(
                        f"{guidance} Details: {type(exc).__name__}: {exc}"
                    ) from exc
    return _model


def reset_tracker() -> None:
    """Reset project-level tracking state before processing a new video."""
    global _next_track_id, _tracker_frame
    with _tracker_lock:
        _tracks.clear()
        _next_track_id = 1
        _tracker_frame = 0


def _box_iou(first: list[int], second: list[int]) -> float:
    """Calculate intersection over union for two xyxy boxes."""
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def _assign_track_ids(detections: list[dict]) -> None:
    """Match detections between frames using greedy class-aware box overlap."""
    global _next_track_id, _tracker_frame
    with _tracker_lock:
        _tracker_frame += 1
        active = {
            track_id: track for track_id, track in _tracks.items()
            if _tracker_frame - track["frame"] <= _TRACK_MAX_AGE
        }
        candidates = []
        for detection_index, detection in enumerate(detections):
            for track_id, track in active.items():
                if detection["label"] != track["label"]:
                    continue
                overlap = _box_iou(detection["box"], track["box"])
                if overlap >= _TRACK_IOU_THRESHOLD:
                    candidates.append((overlap, detection_index, track_id))

        matched_detections = set()
        matched_tracks = set()
        for _overlap, detection_index, track_id in sorted(candidates, reverse=True):
            if detection_index in matched_detections or track_id in matched_tracks:
                continue
            detections[detection_index]["track_id"] = track_id
            matched_detections.add(detection_index)
            matched_tracks.add(track_id)

        for detection_index, detection in enumerate(detections):
            if detection_index not in matched_detections:
                detection["track_id"] = _next_track_id
                _next_track_id += 1
            _tracks[detection["track_id"]] = {
                "label": detection["label"], "box": detection["box"],
                "frame": _tracker_frame,
            }

        for track_id in tuple(_tracks):
            if _tracker_frame - _tracks[track_id]["frame"] > _TRACK_MAX_AGE:
                del _tracks[track_id]


def detect_vehicles(frame: np.ndarray, track: bool = False) -> tuple[np.ndarray, list[dict]]:
    """Detect vehicles, optionally assigning persistent IDs across video frames."""
    model = get_model()
    results = model.predict(frame, classes=list(VEHICLE_CLASSES), verbose=False)
    annotated = frame.copy()
    detections = []
    if not results or results[0].boxes is None:
        if track:
            _assign_track_ids(detections)
        return annotated, detections

    for box in results[0].boxes:
        class_id = int(box.cls[0].item())
        label = VEHICLE_CLASSES.get(class_id)
        if label is None:
            continue
        confidence = float(box.conf[0].item())
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        detections.append({"label": label, "confidence": round(confidence, 4),
                           "track_id": None,
                           "box": [x1, y1, x2, y2]})

    if track:
        _assign_track_ids(detections)

    for detection in detections:
        label = detection["label"]
        confidence = detection["confidence"]
        track_id = detection["track_id"]
        x1, y1, x2, y2 = detection["box"]
        color = (57, 220, 145)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        identity = f" ID:{track_id}" if track_id is not None else ""
        text = f"{label}{identity} {confidence * 100:.1f}%"
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
        )
        top = max(0, y1 - text_height - baseline - 7)
        cv2.rectangle(annotated, (x1, top), (x1 + text_width + 10, y1), color, -1)
        cv2.putText(annotated, text, (x1 + 5, max(text_height, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (12, 25, 25), 2, cv2.LINE_AA)
    return annotated, detections
