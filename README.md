# VisionPort

VisionPort is a terminal-based Indian traffic vehicle detection application using Python, OpenCV, and YOLOv8n. It detects cars, motorcycles, buses, and trucks (COCO classes 2, 3, 5, and 7).

## Run in Windows with VS Code

1. Open the `VisionPort` folder in VS Code and choose **Terminal → New Terminal**.
2. Activate the project environment, or use its Python executable directly:

   ```powershell
   .\venv\Scripts\Activate.ps1
   python main.py
   ```

3. Choose an option from the terminal menu:
   - **1** opens the webcam preview. Press **C** or **Space** to capture and analyze one frame. Press **Q** or **Esc** to cancel. Press any key in the annotated result window to return to the menu.
   - **2** lists supported videos in `data/videos/`. Select a listed number or enter a filename. The video plays unmodified in an OpenCV window. Press **Q** to stop playback and return to the menu.
   - **3** lists videos and extracts every Nth frame into `data/frames/<video-name>/`. Press Enter at the interval prompt to save every frame.
   - **4** lists videos and plays the selected video with YOLO detections and tracking IDs (for example, `car ID:3`). Click two points with a known real-world distance in the video, then enter that distance in meters to show approximate speeds. Press **Esc** during calibration to skip speed estimates, or **Q** during playback to stop. IDs reset for each video.
   - **5** monitors a signal junction in a selected video. Draw one box around the traffic queue and another tightly around the signal light. It estimates red/yellow/green status, shows stopped tracked vehicles' waiting time during red, and alerts when the default queue threshold of 8 vehicles remains exceeded for about 2 seconds. Press **Q** to stop.
   - **6** exits.

If PowerShell blocks environment activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate again. In VS Code, use **Ctrl+Shift+P → Python: Select Interpreter** to select `venv`.

## Dependencies and model

Install dependencies in the active environment with `python -m pip install -r requirements.txt`. Put videos in `data/videos/` (`.mp4`, `.avi`, `.mov`, `.mkv`, `.m4v`, or `.wmv`). YOLOv8n loads once and is reused for each inference. Ultralytics downloads `yolov8n.pt` on first use if the weights are not already in the project folder, so internet access is needed for that first download. A webcam must be connected for option 1. OpenCV requires a working desktop display for its preview windows.

## Detection modules

- `main.py` provides the interactive terminal menu and camera/video playback loops.
- `vehicle_detection.py` loads YOLOv8n, annotates vehicle detections, and assigns persistent IDs by matching same-class boxes across frames. Video speed estimates use calibrated pixel displacement and frame rate; perspective and road-plane differences can make them inaccurate, especially away from the calibration area.
- `traffic_analytics.py` calculates vehicle counts and traffic density for each displayed frame. The junction monitor uses the selected signal-light region's visible color to estimate red/yellow/green status and tracks stopped vehicle IDs during red to estimate their waiting time. These estimates depend on visible signal colors, camera position, and tracking quality.
- `camera.py`, `image_processing.py`, `video_processing.py`, and `static/` contain the earlier capture, processing, API, and dashboard helpers. The default launch command is now `python main.py` and opens the terminal app.
