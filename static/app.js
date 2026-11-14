(() => {
  const form = document.querySelector('#upload-form');
  const input = document.querySelector('#file-input');
  const dropzone = document.querySelector('#dropzone');
  const analyzeButton = document.querySelector('#analyze-button');
  const selectedFile = document.querySelector('#selected-file');
  const fileName = document.querySelector('#file-name');
  const toast = document.querySelector('#toast');
  const cameraPanel = document.querySelector('#camera-panel');
  const cameraVideo = document.querySelector('#camera-video');
  const cameraCanvas = document.querySelector('#camera-canvas');
  const cameraMessage = document.querySelector('#camera-message');
  const startCameraButton = document.querySelector('#start-camera');
  const captureButton = document.querySelector('#capture-image');
  const stopCameraButton = document.querySelector('#stop-camera');
  let cameraStream = null;
  let mode = 'image';
  let file = null;

  const elements = Object.fromEntries(['empty-state','loading-state','error-state','result-content','result-state',
    'error-message','image-preview-wrap','image-preview','image-download-link','video-preview-wrap','video-preview','download-link',
    'total-count','scope-label','density','density-foot','frame-meta','detection-total','detection-list','detections-section',
    'loading-title','loading-copy'].map(id => [id, document.getElementById(id)]));

  document.querySelectorAll('.tab').forEach(tab => tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(item => {
      const active = item === tab;
      item.classList.toggle('active', active);
      item.setAttribute('aria-selected', String(active));
    });
    mode = tab.dataset.mode;
    const cameraMode = mode === 'camera';
    form.hidden = cameraMode;
    cameraPanel.hidden = !cameraMode;
    if (!cameraMode) stopCamera();
    input.accept = mode === 'image' ? 'image/jpeg,image/png,image/webp,image/bmp' : 'video/mp4,video/avi,video/quicktime,video/x-matroska,.m4v';
    input.value = '';
    setFile(null);
    document.querySelector('#file-hint').textContent = mode === 'image' ? 'JPG, PNG, WEBP up to 100 MB' : 'MP4, AVI, MOV, MKV up to 100 MB';
    document.querySelector('.upload-glyph').textContent = mode === 'image' ? '↑' : '▷';
  }));

  function setFile(nextFile) {
    file = nextFile;
    selectedFile.hidden = !file;
    dropzone.hidden = Boolean(file);
    fileName.textContent = file?.name || '';
    analyzeButton.disabled = !file;
  }
  input.addEventListener('change', () => {
    const chosen = input.files[0];
    if (chosen) acceptFile(chosen);
    else setFile(null);
  });
  document.querySelector('#clear-file').addEventListener('click', () => { input.value = ''; setFile(null); });
  dropzone.addEventListener('dragover', event => { event.preventDefault(); dropzone.classList.add('dragover'); });
  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
  dropzone.addEventListener('drop', event => {
    event.preventDefault(); dropzone.classList.remove('dragover');
    const dropped = event.dataTransfer.files[0];
    if (dropped) acceptFile(dropped);
  });

  function acceptFile(candidate) {
    const allowed = mode === 'image' ? /\.(jpe?g|png|webp|bmp)$/i : /\.(mp4|avi|mov|mkv|m4v)$/i;
    if (!allowed.test(candidate.name)) { showToast(`Choose a supported ${mode} file.`); return; }
    if (candidate.size > 100 * 1024 * 1024) { showToast('File is larger than 100 MB.'); return; }
    setFile(candidate);
  }
  function showToast(message) {
    toast.textContent = message;
    clearTimeout(showToast.timer);
    showToast.timer = setTimeout(() => { toast.textContent = ''; }, 4200);
  }
  function stopCamera() {
    if (cameraStream) cameraStream.getTracks().forEach(track => track.stop());
    cameraStream = null;
    cameraVideo.srcObject = null;
    captureButton.disabled = true;
    stopCameraButton.disabled = true;
    startCameraButton.disabled = false;
    document.querySelector('#camera-placeholder').hidden = false;
  }
  startCameraButton.addEventListener('click', async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      cameraMessage.textContent = 'Camera access is unavailable. Open VisionPort in a supported browser using localhost or HTTPS.';
      return;
    }
    startCameraButton.disabled = true;
    cameraMessage.textContent = 'Requesting camera access…';
    try {
      cameraStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false });
      cameraVideo.srcObject = cameraStream;
      await cameraVideo.play();
      document.querySelector('#camera-placeholder').hidden = true;
      captureButton.disabled = false;
      stopCameraButton.disabled = false;
      cameraMessage.textContent = 'Camera is ready. Position the scene and capture a frame.';
    } catch (error) {
      stopCamera();
      cameraMessage.textContent = error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError'
        ? 'Camera permission was denied. Allow camera access in your browser settings, then try again.'
        : error.name === 'NotFoundError' || error.name === 'DevicesNotFoundError'
          ? 'No camera was found. Connect or enable a webcam, then try again.'
          : 'Could not start the camera. Check that it is connected and not in use by another app.';
    }
  });
  stopCameraButton.addEventListener('click', () => {
    stopCamera();
    cameraMessage.textContent = 'Camera stopped.';
  });
  captureButton.addEventListener('click', async () => {
    if (!cameraStream || !cameraVideo.videoWidth) {
      cameraMessage.textContent = 'Wait for the camera preview to become ready, then capture again.';
      return;
    }
    cameraCanvas.width = cameraVideo.videoWidth;
    cameraCanvas.height = cameraVideo.videoHeight;
    cameraCanvas.getContext('2d').drawImage(cameraVideo, 0, 0, cameraCanvas.width, cameraCanvas.height);
    captureButton.disabled = true;
    elements['loading-title'].textContent = 'Analyzing your camera frame';
    elements['loading-copy'].textContent = 'YOLO is identifying vehicles in the captured image.';
    showState('loading');
    cameraMessage.textContent = 'Frame captured. Running vehicle analysis…';
    try {
      const blob = await new Promise((resolve, reject) => cameraCanvas.toBlob(result => result ? resolve(result) : reject(new Error('Could not capture a camera frame.')), 'image/jpeg', 0.92));
      const payload = new FormData();
      payload.append('file', blob, 'camera-capture.jpg');
      const response = await fetch('/api/detect/frame', { method: 'POST', body: payload });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'The server could not analyze this frame.');
      paintStats(data, false);
      cameraMessage.textContent = 'Analysis complete. Capture another frame whenever you are ready.';
    } catch (error) {
      elements['error-message'].textContent = error.message || 'Could not analyze the captured frame.';
      showState('error');
      cameraMessage.textContent = 'Capture analysis failed. The camera is still available; try again.';
    } finally {
      captureButton.disabled = !cameraStream;
    }
  });
  function showState(state) {
    elements['empty-state'].hidden = state !== 'empty';
    elements['loading-state'].hidden = state !== 'loading';
    elements['error-state'].hidden = state !== 'error';
    elements['result-content'].hidden = state !== 'result';
    elements['result-state'].classList.toggle('done', state === 'result');
    elements['result-state'].innerHTML = state === 'result' ? '<i></i> ANALYSIS COMPLETE' : state === 'loading' ? '<i></i> PROCESSING' : '<i></i> AWAITING INPUT';
  }
  function paintStats(data, isVideo) {
    document.querySelector('#total-count').textContent = Number(data.total || 0).toLocaleString();
    document.querySelector('#scope-label').textContent = isVideo ? 'video' : 'image';
    const density = document.querySelector('#density');
    density.textContent = data.density || 'LOW';
    density.className = `density-value ${(data.density || 'low').toLowerCase()}`;
    elements['density-foot'].textContent = isVideo
      ? `Average ${data.average_vehicles_per_frame ?? 0} vehicles per frame`
      : 'Based on this frame';
    for (const name of ['car','motorcycle','bus','truck']) {
      document.querySelector(`#count-${name}`).textContent = Number(data.counts?.[name] || 0).toLocaleString();
    }
    elements['frame-meta'].textContent = isVideo ? `${Number(data.frames_processed).toLocaleString()} FRAMES` : '';
    const detections = data.detections || [];
    elements['detection-total'].textContent = `${detections.length} OBJECT${detections.length === 1 ? '' : 'S'}`;
    elements['detection-list'].replaceChildren();
    detections.forEach(item => {
      const chip = document.createElement('span');
      chip.className = 'detection-chip';
      chip.textContent = `${item.label} · ${(item.confidence * 100).toFixed(1)}%`;
      elements['detection-list'].append(chip);
    });
    elements['detections-section'].hidden = isVideo;
    elements['image-preview-wrap'].hidden = isVideo;
    elements['video-preview-wrap'].hidden = !isVideo;
    if (isVideo) {
      elements['video-preview'].src = data.video_url;
      elements['download-link'].href = data.download_url;
    } else {
      elements['image-preview'].src = data.image;
      elements['image-download-link'].href = `${data.image_url}/download`;
    }
    showState('result');
  }

  form.addEventListener('submit', async event => {
    event.preventDefault();
    if (!file) return;
    analyzeButton.disabled = true;
    elements['loading-title'].textContent = mode === 'video' ? 'Analyzing your video' : 'Analyzing your image';
    elements['loading-copy'].textContent = mode === 'video' ? 'Processing each frame and building your annotated video.' : 'YOLO is identifying vehicles in your image.';
    showState('loading');
    const payload = new FormData();
    payload.append('file', file);
    try {
      const response = await fetch(`/api/detect/${mode}`, { method: 'POST', body: payload });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'The server could not process this file.');
      paintStats(data, mode === 'video');
    } catch (error) {
      elements['error-message'].textContent = error.message || 'Could not connect to VisionPort. Check that the server is running.';
      showState('error');
    } finally {
      analyzeButton.disabled = !file;
    }
  });
})();
