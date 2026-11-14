"""Interactive terminal application for VisionPort traffic detection."""

from pathlib import Path
from math import hypot
from collections import Counter, deque

import cv2

import traffic_analytics
import vehicle_detection

WINDOW_TITLE = "VisionPort - Vehicle Detection"
VIDEOS_DIR = Path(__file__).resolve().parent / "data" / "videos"
FRAMES_DIR = Path(__file__).resolve().parent / "data" / "frames"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv"}


def capture_and_analyze_webcam(camera_index: int = 0) -> None:
    """Preview the webcam, capture one frame on C/space, and show detections."""
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        camera.release()
        print("Could not open the webcam. Check that it is connected and available.")
        return

    frame_to_analyze = None
    print("Webcam ready. Press C or Space to capture, or Q/Esc to return to the menu.")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Could not read a frame from the webcam.")
                return
            cv2.imshow("VisionPort - Webcam (C/Space to capture, Q to cancel)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("c"), ord(" ")):
                frame_to_analyze = frame.copy()
                break
            if key in (ord("q"), 27):
                print("Camera capture cancelled.")
                return
    finally:
        camera.release()
        cv2.destroyAllWindows()

    try:
        annotated, detections = vehicle_detection.detect_vehicles(frame_to_analyze)
        stats = traffic_analytics.frame_statistics(detections)
        cv2.putText(annotated, f"Vehicles: {stats['total']}  Density: {stats['density']}",
                    (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (70, 240, 190), 2,
                    cv2.LINE_AA)
        cv2.imshow(WINDOW_TITLE + " - Captured image (press any key to close)", annotated)
        cv2.waitKey(0)
    except Exception as exc:
        print(f"Could not analyze the captured frame: {exc}")
    finally:
        cv2.destroyAllWindows()


def play_video(video_path: str | Path) -> bool:
    """Play a video without detection; press Q to stop playback."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        return False

    fps = capture.get(cv2.CAP_PROP_FPS)
    delay_ms = max(1, round(1000 / fps)) if fps and fps > 0 else 33
    frame_number = 0
    window_name = WINDOW_TITLE + " - Video Playback (Q to stop)"
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_number += 1
            cv2.imshow(window_name, frame)
            if cv2.waitKey(delay_ms) & 0xFF == ord("q"):
                break
        return frame_number > 0
    finally:
        capture.release()
        cv2.destroyAllWindows()


def choose_video() -> Path | None:
    """List videos in data/videos and return the selected file, if any."""
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    videos = sorted(
        (path for path in VIDEOS_DIR.iterdir()
         if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS),
        key=lambda path: path.name.lower(),
    )
    if not videos:
        print(f"No supported videos found in {VIDEOS_DIR}.")
        print("Add an MP4, AVI, MOV, MKV, M4V, or WMV file and try again.")
        return None

    print(f"\nVideos in {VIDEOS_DIR}:")
    for index, path in enumerate(videos, start=1):
        print(f"  {index}. {path.name}")
    selection = input("Enter a number or video filename (Q to cancel): ").strip().strip('"')
    if selection.lower() == "q":
        return None

    if selection.isdigit() and 1 <= int(selection) <= len(videos):
        selected = videos[int(selection) - 1]
    else:
        candidate = (VIDEOS_DIR / selection).resolve()
        if candidate.parent != VIDEOS_DIR.resolve() or not candidate.is_file() or candidate.suffix.lower() not in VIDEO_EXTENSIONS:
            print("Choose one of the listed videos by number or enter a filename from data/videos.")
            return None
        selected = candidate
    return selected


def choose_and_play_video() -> None:
    """Select a video from data/videos and play it without detection."""
    selected = choose_video()
    if selected is None:
        return

    if not play_video(selected):
        print("OpenCV could not read this video or it contains no readable frames.")


def calibrate_speed_scale(frame) -> float | None:
    """Ask for a known road distance to estimate meters represented by a pixel."""
    points = []
    preview = frame.copy()
    window = WINDOW_TITLE + " - Speed calibration (click two points, Esc to skip)"

    def on_click(event, px, py, _flags, _data):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((px, py))
            cv2.circle(preview, (px, py), 6, (0, 255, 255), -1)
            if len(points) == 2:
                cv2.line(preview, points[0], points[1], (0, 255, 255), 2)

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_click)
    while len(points) < 2:
        canvas = preview.copy()
        cv2.putText(canvas, "Click two points with a known distance; Esc skips speed.",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2,
                    cv2.LINE_AA)
        cv2.imshow(window, canvas)
        if cv2.waitKey(20) & 0xFF == 27:
            cv2.destroyWindow(window)
            return None
    cv2.imshow(window, preview)
    cv2.waitKey(200)
    cv2.destroyWindow(window)

    pixel_distance = hypot(points[1][0] - points[0][0],
                           points[1][1] - points[0][1])
    if pixel_distance <= 0:
        return None
    try:
        meters = float(input("Real-world distance between those points (meters): ").strip())
        if meters <= 0:
            raise ValueError
    except ValueError:
        print("Invalid distance; speed will be omitted. Vehicle tracking will continue.")
        return None
    return meters / pixel_distance


def detect_video(video_path: str | Path) -> bool:
    """Play a video with vehicle IDs and calibrated approximate speed estimates."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        return False

    fps = capture.get(cv2.CAP_PROP_FPS)
    delay_ms = max(1, round(1000 / fps)) if fps and fps > 0 else 33
    fps = fps if fps and fps > 0 else 30.0
    ok, first_frame = capture.read()
    if not ok:
        capture.release()
        return False
    print("Speed estimates are approximate and depend on camera perspective and calibration.")
    meters_per_pixel = calibrate_speed_scale(first_frame)
    frame_number = 0
    window_name = WINDOW_TITLE + " - YOLO Detection (Q to stop)"
    vehicle_detection.reset_tracker()
    previous_positions = {}
    speed_history = {}

    def process_frame(frame):
        nonlocal frame_number
        annotated, detections = vehicle_detection.detect_vehicles(frame, track=True)
        for detection in detections:
            track_id = detection.get("track_id")
            if track_id is None:
                continue
            x1, y1, x2, y2 = detection["box"]
            position = ((x1 + x2) / 2, y2)
            previous = previous_positions.get(track_id)
            if meters_per_pixel is not None and previous is not None:
                old_position, old_frame = previous
                elapsed = (frame_number - old_frame) / fps
                if elapsed > 0:
                    distance = hypot(position[0] - old_position[0],
                                     position[1] - old_position[1])
                    current_speed = distance * meters_per_pixel / elapsed * 3.6
                    history = speed_history.setdefault(track_id, [])
                    history.append(current_speed)
                    del history[:-5]
                    speed = sum(history) / len(history)
                    cv2.putText(annotated, f"~{speed:.1f} km/h",
                                (x1, min(annotated.shape[0] - 8, y2 + 22)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2,
                                cv2.LINE_AA)
            previous_positions[track_id] = (position, frame_number)
        stats = traffic_analytics.frame_statistics(detections)
        frame_number += 1
        cv2.putText(annotated,
                    f"Frame: {frame_number}  Vehicles: {stats['total']}  Density: {stats['density']}",
                    (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (70, 240, 190), 2,
                    cv2.LINE_AA)
        cv2.imshow(window_name, annotated)

    try:
        process_frame(first_frame)
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            process_frame(frame)
            if cv2.waitKey(delay_ms) & 0xFF == ord("q"):
                break
        return frame_number > 0
    finally:
        capture.release()
        cv2.destroyAllWindows()


def choose_and_detect_video() -> None:
    """Select a video from data/videos and play it with YOLO detections."""
    selected = choose_video()
    if selected is None:
        return
    if not detect_video(selected):
        print("OpenCV could not read this video or it contains no readable frames.")


def detect_junction_congestion(video_path: str | Path, congestion_threshold: int = 8,
                               alert_seconds: float = 2.0) -> bool:
    """Monitor a user-selected junction area and alert on sustained queues."""
    if congestion_threshold < 1 or alert_seconds <= 0:
        raise ValueError("The vehicle threshold and alert duration must be positive.")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        return False

    ok, first_frame = capture.read()
    if not ok:
        capture.release()
        return False

    roi = cv2.selectROI(
        WINDOW_TITLE + " - Select traffic queue area, then press Enter",
        first_frame, showCrosshair=True, fromCenter=False,
    )
    cv2.destroyAllWindows()
    x, y, width, height = map(int, roi)
    if width == 0 or height == 0:
        capture.release()
        print("Junction selection cancelled.")
        return True

    signal_roi = cv2.selectROI(
        WINDOW_TITLE + " - Select traffic light, then press Enter",
        first_frame, showCrosshair=True, fromCenter=False,
    )
    cv2.destroyAllWindows()
    sx, sy, sw, sh = map(int, signal_roi)
    if sw == 0 or sh == 0:
        capture.release()
        print("Traffic light selection cancelled.")
        return True

    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 30.0
    delay_ms = max(1, round(1000 / fps))
    required_frames = max(1, round(alert_seconds * fps))
    high_frames = 0
    alert_active = False
    frame_number = 0
    signal_history = deque(maxlen=5)
    previous_positions = {}
    waiting_since = {}
    window_name = WINDOW_TITLE + " - Junction Congestion Alerts (Q to stop)"
    vehicle_detection.reset_tracker()

    def classify_signal(frame):
        light = frame[sy:sy + sh, sx:sx + sw]
        if light.size == 0:
            return "UNKNOWN"
        hsv = cv2.cvtColor(light, cv2.COLOR_BGR2HSV)
        saturated = hsv[:, :, 1] > 90
        bright = hsv[:, :, 2] > 100
        hue = hsv[:, :, 0]
        ratios = {
            "RED": float((((hue <= 10) | (hue >= 170)) & saturated & bright).mean()),
            "YELLOW": float(((hue >= 15) & (hue <= 38) & saturated & bright).mean()),
            "GREEN": float(((hue >= 39) & (hue <= 90) & saturated & bright).mean()),
        }
        status, ratio = max(ratios.items(), key=lambda pair: pair[1])
        return status if ratio >= 0.04 else "UNKNOWN"

    def analyze_frame(frame):
        nonlocal high_frames, alert_active, frame_number
        annotated, detections = vehicle_detection.detect_vehicles(frame, track=True)
        roi_right, roi_bottom = x + width, y + height
        vehicles_in_zone = sum(
            1 for item in detections
            if x <= (item["box"][0] + item["box"][2]) / 2 < roi_right
            and y <= (item["box"][1] + item["box"][3]) / 2 < roi_bottom
        )
        congested = vehicles_in_zone >= congestion_threshold
        high_frames = high_frames + 1 if congested else 0
        alert = high_frames >= required_frames
        if alert and not alert_active:
            print(f"CONGESTION ALERT: {vehicles_in_zone} vehicles in the selected junction area.")
        alert_active = alert

        signal_history.append(classify_signal(frame))
        signal = Counter(signal_history).most_common(1)[0][0]
        current_time = frame_number / fps
        visible_ids = set()
        for item in detections:
            track_id = item.get("track_id")
            if track_id is None:
                continue
            visible_ids.add(track_id)
            bx1, by1, bx2, by2 = item["box"]
            position = ((bx1 + bx2) / 2, by2)
            previous = previous_positions.get(track_id)
            stopped = False
            if previous is not None:
                old_position, old_frame = previous
                elapsed_frames = max(1, frame_number - old_frame)
                displacement = hypot(position[0] - old_position[0],
                                     position[1] - old_position[1])
                stopped = displacement / elapsed_frames <= 1.5
            previous_positions[track_id] = (position, frame_number)

            inside_queue = x <= position[0] < roi_right and y <= position[1] < roi_bottom
            if signal == "RED" and inside_queue and stopped:
                waiting_since.setdefault(track_id, current_time)
                waited = current_time - waiting_since[track_id]
                cv2.putText(annotated, f"Wait {waited:.1f}s", (bx1, max(18, by1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                waiting_since.pop(track_id, None)
        for track_id in waiting_since.keys() - visible_ids:
            waiting_since.pop(track_id, None)

        color = (0, 0, 255) if alert else (0, 190, 255) if congested else (57, 220, 145)
        cv2.rectangle(annotated, (x, y), (roi_right, roi_bottom), color, 3)
        cv2.rectangle(annotated, (sx, sy), (sx + sw, sy + sh), (255, 190, 0), 2)
        status = "CONGESTION ALERT" if alert else "Queue building" if congested else "Traffic clear"
        longest_wait = max((current_time - start for start in waiting_since.values()), default=0)
        cv2.putText(annotated,
                    f"Signal: {signal} | {status} | Vehicles: {vehicles_in_zone} | Max wait: {longest_wait:.1f}s",
                    (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        frame_number += 1
        cv2.imshow(window_name, annotated)

    try:
        analyze_frame(first_frame)
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            analyze_frame(frame)
            if cv2.waitKey(delay_ms) & 0xFF == ord("q"):
                break
        return frame_number > 0
    finally:
        capture.release()
        cv2.destroyAllWindows()


def choose_and_monitor_junction() -> None:
    """Choose a video and monitor a manually selected signal junction ROI."""
    selected = choose_video()
    if selected is None:
        return
    threshold_text = input("Alert at how many vehicles in the junction area? [8]: ").strip()
    try:
        threshold = int(threshold_text) if threshold_text else 8
        if threshold < 1:
            raise ValueError
    except ValueError:
        print("Enter a whole number greater than zero.")
        return
    if not detect_junction_congestion(selected, threshold):
        print("OpenCV could not read this video or it contains no readable frames.")


def extract_frames(video_path: str | Path, frame_interval: int = 1) -> int:
    """Save every Nth frame from a video into data/frames/<video name>/."""
    if frame_interval < 1:
        raise ValueError("Frame interval must be at least 1.")

    video_path = Path(video_path)
    output_dir = FRAMES_DIR / video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        print(f"Could not open video file: {video_path}")
        return 0

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    print(f"\nExtracting frames from: {video_path.name}")
    print(f"Total frames: {total_frames} | FPS: {fps:.2f}")
    print(f"Saving frames to: {output_dir}")

    frame_count = 0
    saved_count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_count += 1

            if frame_count % frame_interval == 0:
                frame_file = output_dir / f"frame_{saved_count:05d}.jpg"
                if not cv2.imwrite(str(frame_file), frame):
                    print(f"Could not save frame: {frame_file}")
                    continue
                saved_count += 1
                if saved_count % 50 == 0:
                    print(f"Processed {frame_count}/{total_frames or '?'} frames ({saved_count} saved).")
    finally:
        capture.release()

    print(f"Extraction complete. Saved {saved_count} frames to {output_dir}.")
    return saved_count


def choose_and_extract_frames() -> None:
    """Select a video and extract frames at a user-selected interval."""
    selected = choose_video()
    if selected is None:
        return
    interval_text = input("Save every how many frames? [1 = every frame]: ").strip()
    try:
        frame_interval = int(interval_text) if interval_text else 1
        if frame_interval < 1:
            raise ValueError
    except ValueError:
        print("Enter a whole number greater than zero.")
        return
    extract_frames(selected, frame_interval)


def show_menu() -> str:
    print("\n=== VisionPort | Indian Traffic Detection ===")
    print("1. Take a picture from the webcam and analyze it")
    print("2. Play a video file without detection")
    print("3. Extract frames from a video")
    print("4. Track vehicles and estimate approximate speed")
    print("5. Detect signal status, congestion, and vehicle waiting times")
    print("6. Exit")
    return input("Choose an option (1-6): ").strip()


def main() -> None:
    while True:
        try:
            choice = show_menu()
            if choice == "1":
                capture_and_analyze_webcam()
            elif choice == "2":
                choose_and_play_video()
            elif choice == "3":
                choose_and_extract_frames()
            elif choice == "4":
                choose_and_detect_video()
            elif choice == "5":
                choose_and_monitor_junction()
            elif choice == "6":
                print("Goodbye.")
                break
            else:
                print("Please choose 1, 2, 3, 4, 5, or 6.")
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break
        except cv2.error as exc:
            print(f"OpenCV error: {exc}")
        except Exception as exc:
            print(f"VisionPort could not continue: {exc}")


if __name__ == "__main__":
    main()
