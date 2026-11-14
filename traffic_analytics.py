"""Vehicle totals and traffic density calculations."""

from collections import Counter
from typing import Iterable

VEHICLE_NAMES = ("car", "motorcycle", "bus", "truck")


def density_for(vehicle_count: float) -> str:
    """Return LOW (0–5), MEDIUM (6–15), or HIGH (16+) for a frame count."""
    if vehicle_count <= 5:
        return "LOW"
    if vehicle_count <= 15:
        return "MEDIUM"
    return "HIGH"


def frame_statistics(detections: Iterable[dict]) -> dict:
    """Summarize one frame's detected-object records."""
    counts = Counter({name: 0 for name in VEHICLE_NAMES})
    records = list(detections)
    counts.update(item["label"] for item in records if item.get("label") in VEHICLE_NAMES)
    total = sum(counts.values())
    return {
        "total": total,
        "counts": dict(counts),
        "density": density_for(total),
        "detections": records,
    }


def video_statistics(category_totals: Counter, frame_count: int, frame_vehicle_total: int) -> dict:
    """Summarize total frame detections and average density for a video."""
    average = frame_vehicle_total / frame_count if frame_count else 0
    counts = {name: int(category_totals.get(name, 0)) for name in VEHICLE_NAMES}
    return {
        "total": sum(counts.values()),
        "counts": counts,
        "density": density_for(average),
        "average_vehicles_per_frame": round(average, 2),
        "frames_processed": frame_count,
    }
