"""Image loading, preview, and saved YOLO detection results."""

from pathlib import Path

import cv2

import traffic_analytics
import vehicle_detection


def read_image(image_path: str | Path):
    """Read a color image; return None when the path is invalid or unsupported."""
    return cv2.imread(str(image_path), cv2.IMREAD_COLOR)


def display_image(image_path: str | Path) -> bool:
    """Show a traffic image in an OpenCV window. Press any key to close."""
    image = read_image(image_path)
    if image is None:
        return False
    cv2.imshow("VisionPort - Traffic image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return True


def process_image(image_path: str | Path, output_path: str | Path) -> dict:
    """Detect vehicles in an image, save the annotated image, and return stats."""
    image = read_image(image_path)
    if image is None:
        raise ValueError("Could not decode this image.")
    annotated, detections = vehicle_detection.detect_vehicles(image)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), annotated):
        raise RuntimeError("Could not save the processed image.")
    stats = traffic_analytics.frame_statistics(detections)
    stats["output_path"] = destination
    return stats
