"""Webcam capture helpers."""

from pathlib import Path

import cv2


def capture_traffic_image(output_path: str | Path, camera_index: int = 0,
                          preview: bool = True) -> Path:
    """Capture a webcam image and save it; optionally preview and press S to save."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError("Could not open the webcam. Check that it is connected and available.")
    try:
        if not preview:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Could not capture a frame from the webcam.")
        else:
            frame = None
            print("Webcam preview: press S to save the image, or Q to cancel.")
            while True:
                ok, current_frame = camera.read()
                if not ok:
                    raise RuntimeError("Could not read a frame from the webcam.")
                cv2.imshow("VisionPort - Camera", current_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("s"):
                    frame = current_frame
                    break
                if key == ord("q"):
                    raise RuntimeError("Camera capture cancelled.")
        if not cv2.imwrite(str(destination), frame):
            raise RuntimeError("Could not save the captured image.")
        return destination
    finally:
        camera.release()
        if preview:
            cv2.destroyAllWindows()
