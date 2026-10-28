# Real-Time Object Detection and Multi-Object Tracking

A complete, runnable computer-vision project that detects objects in webcam or
video streams with **YOLO11**, assigns persistent identities with **ByteTrack**
or **BoT-SORT**, and presents live statistics and an annotated, downloadable
video in a **Streamlit** dashboard.

The project uses your existing `models/yolo11n.pt` weights. It does **not**
train a model or download a dataset.

---

## Abstract

This project implements a real-time object detection and tracking pipeline for
video streams. Frames are captured with OpenCV and passed to an Ultralytics
YOLO11 detector. The detector's bounding boxes are associated across frames by
a multi-object tracker (ByteTrack or BoT-SORT) using Ultralytics' persistent
tracking mode. Each detection is annotated with its class label, confidence
score and tracker-assigned ID. A background worker thread decouples video
processing from the web interface, allowing the Streamlit dashboard to display
live frames, FPS, object counts by class, unique session IDs and per-ID
observation statistics without blocking on inference. Annotated results are
written to an MP4 file at the source resolution and frame rate and can be
downloaded when processing finishes. The implementation is modular, handles
camera and file errors gracefully, releases all media resources, and includes
unit, integration and end-to-end verification suites.

---

## 1. Objectives

1. Detect objects on every processed frame with class names and confidence scores.
2. Maintain persistent tracking IDs across consecutive frames.
3. Display live object, class, FPS, frame and session statistics.
4. Support both webcam and uploaded/existing video sources.
5. Preserve source resolution and frame rate in annotated output where possible.
6. Provide pause, resume, stop and download controls.
7. Fail gracefully with actionable messages instead of raw tracebacks.
8. Use CUDA automatically when available and fall back to CPU with a warning.
9. Keep the code modular, testable and documented for academic demonstration.

### Scope

- **In scope:** inference with existing YOLO detection weights, tracking,
  annotation, live dashboard, MP4 output, diagnostics, tests.
- **Out of scope:** model training, dataset annotation, custom class training,
  distributed processing, audio preservation, multi-camera synchronization.

---

## 2. Key features

| Area | Implemented behaviour |
|---|---|
| Detection | YOLO11 via Ultralytics, configurable confidence/IoU, class filter, inference size |
| Tracking | ByteTrack or BoT-SORT with `persist=True`; IDs assigned only by the tracker |
| Input | Webcam index (DirectShow/Media Foundation fallbacks on Windows), uploaded MP4/AVI/MOV/MKV, pre-existing videos |
| Live output | Annotated frames in the browser, refreshed by a full-page Streamlit rerun every 250 ms |
| Statistics | Current object count, unique session IDs, current/unique class counts, processing and inference FPS, frame count, resolution, status |
| ID registry | First frame, last frame, class, and number of frames observed for each tracker ID |
| Controls | Start, pause, resume, stop; confidence, IoU, tracker, inference size, classes, detection interval, overlay, frame limit, playback throttle |
| Output | Annotated MP4 at source resolution/FPS, verified after writing, downloadable from the dashboard |
| Robustness | Background worker, thread-safe state, graceful errors, guaranteed capture/writer release, no fabricated tracker IDs |
| Device | Automatic CUDA selection, explicit CPU fallback warning, FP16 quantization on CUDA |
| Verification | 34 automated tests plus `verification.py` end-to-end checks |

---

## 3. Methodology

### 3.1 Detection

YOLO11 is a single-stage object detector. The bundled `yolo11n.pt` is the
nano variant of the YOLO11 family and has approximately 2.6 million parameters.
It is pretrained on the 80-class COCO dataset. For a frame \(I_t\), the model
produces a set of boxes

\[
D_t = \{(b_i, s_i, c_i)\}_{i=1}^{N_t}
\]

where \(b_i=(x_1,y_1,x_2,y_2)\) is the bounding box, \(s_i\in[0,1]\) is the
confidence score, and \(c_i\) is the predicted class. Ultralytics performs
letterboxing, inference and non-maximum suppression internally. The confidence
threshold filters low-score boxes before tracking; the IoU threshold controls
NMS overlap.

### 3.2 Multi-object tracking

Tracking converts per-frame detections into temporally consistent identities.
The project delegates association to Ultralytics rather than implementing a
tracker:

- **ByteTrack** (`bytetrack.yaml`) associates high-confidence detections first,
  then uses lower-confidence boxes to recover partially occluded tracks. It is
  the default because it is fast and works well on CPU.
- **BoT-SORT** (`botsort.yaml`) adds camera-motion compensation and improved
  association. The default Ultralytics configuration has ReID disabled, so no
  additional appearance model is downloaded. It can be slower on CPU.

`YOLODetector.track()` calls `model.track(..., persist=True, tracker=...)`.
`persist=True` keeps tracker state between calls, which is what allows an ID to
survive from one frame to the next. A new `TrackingSession` wrapper and
`detector.reset_tracking()` are created at the start of each session, so IDs are
session-scoped and do not leak from one run into another.

> **Important:** the application never invents an ID. If the tracker returns a
> detection without an ID, it is drawn without an ID and excluded from the
> unique-ID registry. It still contributes to the current object and class
> counts.

### 3.3 Detection interval

When the detection interval is \(k>1\), inference runs on frames
\(1, 1+k, 1+2k,\dots\). Intermediate frames reuse the most recent boxes, which
keeps the display smooth on slow hardware. Tracker state and ID statistics are
updated only on inference frames. This is a deliberate latency/throughput
trade-off: IDs are not advanced on skipped frames, so very fast motion can
cause identity switches.

### 3.4 Threaded processing

Streamlit reruns the script for interaction. Running inference directly in the
script would block the UI. Instead, `VideoProcessor` is a daemon thread that:

1. opens the capture and reads the first frame,
2. creates the output writer using the decoded frame size and source FPS,
3. loops over frames, running detection/tracking and drawing annotations,
4. publishes an immutable snapshot of state for the UI,
5. releases the capture and writer, verifies the output, and publishes the
   terminal status.

While a session is active, the page polls `processor.snapshot()` and schedules
a full-script rerun every 250 ms. A full rerun (rather than a Streamlit
fragment) keeps the sidebar status and button states synchronized with the
worker and avoids fragment-rerun races at the terminal transition. Processing
itself never waits for the UI. All shared state is protected by a re-entrant
lock, and pause/stop are implemented with `threading.Event` objects.

### 3.5 Statistics semantics

| Metric | Definition |
|---|---|
| Objects now | Number of detections in the latest **inference** frame (including untracked detections) |
| Unique tracking IDs | Number of distinct tracker IDs observed in the current session |
| Objects by class | Class counts from the latest inference frame |
| Unique IDs by class | Distinct IDs grouped by the class seen when the ID first appeared |
| Untracked objects | Current detections that have no tracker ID |
| Processing FPS | Exponential moving average of wall-clock loop throughput (detection + drawing + writing) |
| Inference FPS | Exponential moving average measured only on inference frames |
| Frames processed | Number of frames written/published since the session started (1-based) |
| Seen (frames) | Number of inference frames in which a given ID was observed |
| Status | `idle`, `starting`, `running`, `paused`, `stopping`, `completed`, `stopped`, or `error` |

Session statistics reset when a new webcam or video session starts. They are
not global or cross-session identifiers.

### 3.6 Output verification

After the writer is released, the project reopens the MP4 with OpenCV and
checks that it exists, is not trivially small, decodes a frame, and matches the
expected resolution. Output FPS is taken from the source when it is in the
valid range \([1,240]\); otherwise a safe 30 FPS is used. A failed verification
turns the session status into an error instead of presenting a broken download.
If processing is stopped early, the partial output is still verified and
offered as a partial result.

---

## 4. System architecture

```text
┌────────────────────┐      ┌────────────────────┐
│ Webcam / video file│      │ Streamlit controls │
│  (OpenCV capture)  │      │  + live dashboard  │
└─────────┬──────────┘      └─────────┬──────────┘
          │ frames                    │ snapshot polling (250 ms)
          ▼                           │
┌─────────────────────────────────────────────────────┐
│                VideoProcessor thread                │
│                                                     │
│  frame → YOLODetector.track(persist=True)           │
│          └─ YOLO11 detection                        │
│          └─ ByteTrack / BoT-SORT association        │
│       → TrackingSession.update()                    │
│       → draw boxes, labels, confidence, #ID         │
│       → OpenCV VideoWriter (MP4)                    │
│       → publish thread-safe snapshot                │
└───────────────────────┬─────────────────────────────┘
                        │ release + verify
                        ▼
              output_videos/*.mp4 (download)
```

Data flow per frame:

```text
capture.read() → optional mirror/resize → detect+track → statistics
→ annotate → write MP4 → convert BGR→RGB → publish to UI
```

---

## 5. Project structure

```text
object detection/
├── app.py                     # Streamlit dashboard and control panel
├── detector.py                # YOLO model loading, detection/tracking wrapper, Detection dataclass
├── tracker.py                 # Session statistics and per-ID registry
├── video_processor.py         # Background pipeline, config, thread-safe state, pause/stop, output verification
├── utils.py                   # Paths, validation, drawing, video I/O, formatting, FPS meter
├── errors.py                  # User-facing exception types
├── verification.py            # End-to-end verification runner and diagnostics
├── requirements.txt           # Runtime and test dependencies
├── .streamlit/config.toml     # Upload limits and theme
├── pytest.ini                 # Test discovery configuration
├── models/
│   └── yolo11n.pt             # Existing YOLO11 nano weights (provided by you)
├── input_videos/
│   ├── sample_traffic.avi     # Included demonstration video (848×478, 30 FPS)
│   └── uploads/               # Saved browser uploads (created at runtime)
├── output_videos/             # Annotated MP4s and verification images (created at runtime)
└── tests/
    ├── conftest.py            # Shared fixtures and synthetic video generation
    ├── test_core.py           # Unit tests (no model required)
    ├── test_integration.py    # Real model, tracking, pipeline and control tests
    └── test_app_smoke.py      # Streamlit AppTest smoke test
```

### Module responsibilities

- **`detector.py`** — resolves model paths, loads the model once, converts
  Ultralytics tensors into immutable `Detection` objects, and resets tracker
  state per session.
- **`tracker.py`** — aggregates detections without fabricating IDs, stores
  first/last frame and observation counts, and produces table-ready rows.
- **`video_processor.py`** — owns the capture, writer and worker lifecycle;
  exposes `pause()`, `resume()`, `stop()`, and `snapshot()`.
- **`utils.py`** — validates media, creates unique output paths, draws stable
  colors and labels, verifies output files, and provides formatting helpers.
- **`app.py`** — renders the dashboard and translates controls into a validated
  `ProcessingConfig`.

---

## 6. Installation

### 6.1 Prerequisites

- Python **3.10–3.12** (verified with 3.11.9)
- A webcam for webcam mode (optional; uploaded video works without one)
- Windows 10/11, Linux, or macOS
- For GPU: a CUDA-capable GPU and a CUDA-enabled PyTorch build

### 6.2 Windows (PowerShell)

```powershell
cd "C:\Users\HP\Downloads\Project\object detection"

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation for the current shell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 6.3 Linux / macOS

```bash
cd "/path/to/object detection"

python3.11 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 6.4 CUDA (optional)

Install the PyTorch build that matches your CUDA driver **before** the other
packages. Use the selector at <https://pytorch.org/get-started/locally/>. For
example, for CUDA 12.4:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Verify the installation:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The app uses CUDA automatically when `torch.cuda.is_available()` is `True`.
On CPU it logs a warning and continues with a lower expected FPS.

### 6.5 Model weights

Place your existing weights at:

```text
models/yolo11n.pt
```

The dashboard also accepts a custom `.pt` path. If the file is missing, model
loading fails with a clear message listing the paths that were searched. No
training step is performed. The default model expects the 80 COCO classes
(person, bicycle, car, motorcycle, airplane, bus, train, truck, and so on).

---

## 7. Running the application

From the project directory with the virtual environment active:

```bash
streamlit run app.py
```

Streamlit opens <http://localhost:8501> in your browser. To select a port or
run on a lab server:

```bash
streamlit run app.py --server.port 8501
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

> A remote Streamlit server cannot access the **viewer's** local webcam. Use
> uploaded video when the app runs on a headless or remote machine.

### 7.1 Quick start with the included sample

1. Open the sidebar.
2. Confirm **1 · Model** shows `yolo11n.pt` loaded and a device.
3. Under **2 · Video source**, select **Upload video**.
4. Leave **Or choose a video from `input_videos/`** on `sample_traffic.avi`.
5. Optionally set **Frame limit** to `300` for a short demonstration.
6. Keep ByteTrack, confidence `0.25`, IoU `0.50`, inference size `640`.
7. Click **▶ Start**.
8. Watch boxes, labels, confidence values and `#ID` tags. The object count,
   unique IDs, FPS and ID registry update live.
9. Wait for **Completed**, then use **Download annotated MP4**.

For the included sample, frame 420 onward contains a vehicle that is tracked
across many consecutive frames; it is a reliable demo window.

---

## 8. Dashboard guide

### 8.1 Sidebar controls

| Control | Meaning |
|---|---|
| YOLO weights / Custom path | Reuse existing weights; a custom path is allowed |
| Device | `Auto` prefers CUDA, else CPU; CPU can be forced; CUDA is only offered when available |
| Reload model | Release the current model and load again |
| Input | Webcam or Upload video |
| Camera index / Scan cameras | Choose a camera; scanning briefly probes indices 0–3 |
| Mirror view | Horizontally flips webcam frames (default on) |
| Upload a video | MP4, AVI, MOV or MKV; saved once under `input_videos/uploads/` |
| Existing video | Pick a file already in `input_videos/` |
| Frame limit | `0` processes the whole video; a positive number stops early |
| Throttle output to source FPS | Delays the loop to preserve playback pace (default off) |
| Confidence threshold | Minimum detection score (0.05–0.95) |
| IoU threshold | NMS overlap threshold (0.10–0.95) |
| Tracker | ByteTrack (fast default) or BoT-SORT |
| Inference size | 416–960 px; larger finds smaller objects but is slower |
| Detect all classes | When off, choose a subset of COCO classes |
| Run detection every N frames | 1–5; intermediate frames reuse boxes |
| Draw statistics overlay | Draws object count, unique IDs and FPS on the video |
| Start / Pause / Resume / Stop | Session lifecycle controls |
| Download annotated MP4 | Appears after a session reaches a terminal state |

### 8.2 Live panels

- **Metrics row** — objects now, unique tracking IDs, processing FPS, frames
  processed, resolution and status.
- **Video panel** — latest annotated frame, progress bar and status message.
- **Objects by class** — current and unique-ID counts per class.
- **Tracking ID registry** — up to 20 most recently seen IDs with class,
  first/last frame and frames observed.
- **Session configuration** — source, tracker, thresholds and detection interval.
- **Processed video** — file name, size, frame count, codec and download button.

### 8.3 Status meanings

| Status | Meaning |
|---|---|
| `Starting` | Opening the source and writing the first frame |
| `Running` | Processing normally |
| `Paused` | Worker is waiting; no new frames are read |
| `Stopping` | Stop requested; resources are being released |
| `Completed` | Source or frame limit finished; output verified |
| `Stopped` | User stopped the session; partial output may be available |
| `Error` | Failure with a user-facing message and optional technical details |

---

## 9. Output specification

- **Container/codec:** MP4 with `mp4v`, falling back to `avc1` or `H264`.
- **Resolution:** decoded source resolution (webcam frames are resized only if
  the camera changes size mid-stream).
- **Frame rate:** source FPS when it is between 1 and 240; otherwise 30 FPS.
- **Content:** boxes, class labels, confidence scores, tracker IDs (`#ID` when
  available), and optional session overlay.
- **Audio:** not preserved (OpenCV writes video frames only).
- **Verification:** file exists, size is reasonable, first frame decodes, and
  resolution matches expectation.
- **Location:** `output_videos/` with a timestamped, collision-free file name.
- **Download:** served by Streamlit only after the session is terminal.

Some browsers cannot play `mp4v` directly in an HTML5 player. The download is a
valid MP4 that works in VLC/desktop players; re-encode to H.264 if browser
playback in another tool is required:

```bash
ffmpeg -i output_videos/result.mp4 -c:v libx264 -pix_fmt yuv420p result_h264.mp4
```

---

## 10. Testing and verification

### 10.1 Automated test suite

```bash
python -m pytest -q
```

Coverage includes:

- configuration validation and normalization,
- tracker statistics, duplicate IDs, untracked detections and reset semantics,
- device resolution and CUDA fallback,
- path sanitization, unique output naming and uploaded-file deduplication,
- video probing, writer round-trip and output verification,
- drawing boxes/labels/overlay without crashes on malformed input,
- real YOLO detection on the sample frame,
- persistent IDs across consecutive frames with ByteTrack,
- BoT-SORT execution,
- end-to-end processor output, resolution/FPS/frame count and cleanup,
- detection-interval inference skipping,
- pause/resume/stop state transitions,
- graceful failure for a missing video,
- Streamlit dashboard smoke test.

Fast unit tests only (no model):

```bash
python -m pytest -q tests/test_core.py
```

### 10.2 End-to-end verification

```bash
python verification.py
```

Optional variants:

```bash
python verification.py --video path/to/video.mp4 --start-frame 0 --frames 100
python verification.py --model models/yolo11n.pt --frames 60
python verification.py --camera 0          # only on a machine with a webcam
```

The runner checks dependencies, model loading, detection, unique-ID semantics,
tracking/output, BoT-SORT, graceful error cleanup, and optionally webcam
cleanup. It writes verification artifacts to `output_videos/`.

### 10.3 Current verification result

On the development machine used for this project (Python 3.11.9, Ultralytics
8.4.189, OpenCV 4.10.0, PyTorch 2.14.1 CPU-only, 848×478 sample at 30 FPS):

- `python -m pytest -q` → **34 passed**
- `python verification.py --camera 0` → **8/8 checks passed, 0 skipped**
- Tracking check → 60 frames, 4 unique IDs, stable ID `#1` observed in all 60
  frames, output verified at 848×478 and 30 FPS
- Webcam check → camera 0 opened through DirectShow at 640×480, processed 20
  frames, completed, and released the capture cleanly
- Measured CPU throughput during the tracking check → approximately **16.1
  processing FPS at `imgsz=640`**

These figures are machine-specific, not a general performance guarantee. On
another machine, run `python verification.py --camera 0` only if an accessible
camera is present; otherwise omit the flag and the webcam check is skipped.

The live browser flow was also exercised manually: upload-mode session,
pause/resume, stop, automatic completion, output download button, and webcam
failure/success paths. Temporary smoke artifacts were removed; verification
artifacts remain in `output_videos/`.

---

## 11. Performance notes

| Factor | Effect |
|---|---|
| Inference size (`imgsz`) | 640 is the default. 416/512 increases FPS; 800/960 improves small-object recall at lower FPS. |
| Device | CUDA is much faster. CUDA inference uses 16-bit quantization (`quantize=16`). |
| Tracker | ByteTrack is lighter than BoT-SORT, especially on CPU. |
| Detection interval | 2–5 reduces inference load but reuses boxes between inferences. |
| Frame limit | Useful for demos; does not change per-frame cost. |
| Output writing | Always on for video sessions; encoding adds overhead, especially at high resolution. |
| UI refresh | The dashboard reruns every 250 ms while processing is active; processing does not wait for the UI. |

CUDA is selected automatically. To force CPU, choose **CPU** in the sidebar.
If CUDA is unavailable, the app warns that inference may run below real time.

---

## 12. Configuration reference

### `ProcessingConfig` (main fields)

| Field | Default | Notes |
|---|---|---|
| `source_type` | required | `"webcam"` or `"video"` |
| `source` | required | Camera index (int) for webcam; path for video |
| `output_path` | `None` | MP4 path; webcam sessions do not save by default |
| `conf` | `0.25` | 0–1 |
| `iou` | `0.50` | 0–1 |
| `classes` | `None` | `None` = all classes; otherwise a non-empty sequence of class IDs |
| `tracker` | `"ByteTrack"` | Also accepts `"BoT-SORT"` and tracker YAML aliases |
| `imgsz` | `640` | At least 32 |
| `detect_interval` | `1` | At least 1 |
| `show_overlay` | `True` | Draw count/ID/FPS panel |
| `mirror` | `False` | Horizontal flip (used by webcam UI) |
| `realtime` | `False` | Throttle video playback to source FPS |
| `max_frames` | `0` | 0 = unlimited / whole source |
| `start_frame` | `0` | Seek position for video files |
| `model_name`, `device_label` | `""` | Display metadata only |

### Streamlit settings (`.streamlit/config.toml`)

- Upload/message limit: 1024 MB
- Light theme matching the dashboard palette
- Usage statistics disabled

---

## 13. Troubleshooting

| Problem | Solution |
|---|---|
| `YOLO weights not found` | Put `yolo11n.pt` in `models/` or enter its absolute path under **Custom path**. |
| `CUDA was requested but is not available` | Install CUDA PyTorch or set Device to **Auto/CPU**. |
| CUDA available but CPU selected | Switch Device to **Auto**; the app reports the resolved device. |
| `Could not open camera index 0` | Close other camera apps, allow camera permission, try another index, or use uploaded video. A remote/headless server cannot see your local webcam. |
| `No objects detected` | Lower confidence, choose a scene with visible COCO objects, or try `imgsz=640`. |
| Very low FPS | Use a GPU, choose CPU-friendly ByteTrack, reduce `imgsz`, set detection interval > 1, or limit frames. |
| Output cannot be created | Install a full `opencv-python` build and check write permissions on `output_videos/`. |
| Download missing | The session has not finished/stopped, or output verification failed; inspect the status message and technical details. |
| MP4 won't play in a browser | The file uses `mp4v`; use VLC or re-encode to H.264 with ffmpeg. |
| Upload rejected | Use MP4/AVI/MOV/MKV and keep the file below the configured upload limit. |

Diagnostics:

```bash
python -c "import torch, ultralytics, cv2, streamlit, lap; print(torch.__version__, torch.cuda.is_available())"
python verification.py
```

---

## 14. Limitations

- **Pretrained classes only.** The COCO model cannot detect custom classes
  without fine-tuning, which is outside this project's scope.
- **Identity switches.** Tracking can lose or swap IDs under heavy occlusion,
  fast motion, camera cuts, or when detection is skipped. The project reports
  what the tracker returns; it does not fabricate continuity.
- **CPU performance.** The verified CPU setup runs below the sample video's
  30 FPS. "Real-time" depends on hardware and settings.
- **Session-scoped IDs.** IDs reset on every new session and are not globally
  unique across sessions.
- **No audio.** The annotated output contains video frames only.
- **Webcam through the server.** Streamlit accesses the camera of the machine
  running the Python process, not the viewer's browser device.
- **Fixed display overlay.** The overlay is drawn into the output video when
  enabled; it is not a separate transparent layer.
- **Codec compatibility.** `mp4v` is broadly available through OpenCV but is not
  the most browser-friendly codec.
- **No re-identification model.** BoT-SORT uses its default configuration
  without appearance ReID, so no extra model is downloaded.
- **Single stream.** One processing session/source at a time per browser
  session.

---

## 15. Ethical and operational considerations

Object detection and tracking can be used in surveillance. Use this project
only with informed consent, lawful authority, and appropriate privacy controls.
Do not deploy it for identity verification, punitive decisions, or
safety-critical functions. Pretrained models can reflect dataset biases and may
perform unevenly across people, environments and object types. Validate
performance on the intended population and conditions before any real-world
use. Camera footage and downloaded videos may contain personal data; store and
retain them responsibly.

---

## 16. Future work

1. Add optional appearance ReID for BoT-SORT and compare identity-switch rates.
2. Evaluate on MOT17/MOT20-style metrics (MOTA, IDF1, HOTA) with labelled data.
3. Add line-crossing, zone and speed analytics.
4. Support multiple simultaneous sources and GPU batching.
5. Add H.264 output through FFmpeg for universal browser playback.
6. Add configurable tracker YAML presets and per-class tracker thresholds.
7. Add model export (ONNX/TensorRT) and a throughput/latency benchmark page.
8. Persist session summaries to CSV/JSON for offline analysis.
9. Add authentication and privacy controls for shared deployments.

---

## 17. References

1. Ultralytics. *YOLO11 Documentation.* <https://docs.ultralytics.com/models/yolo11/>
2. Ultralytics. *Track Mode (ByteTrack and BoT-SORT).* <https://docs.ultralytics.com/modes/track/>
3. Jocher, G., Chaurasia, A., & Qiu, J. (2023). *Ultralytics YOLOv8.*
   <https://github.com/ultralytics/ultralytics>
4. Zhang, Y., Sun, P., Jiang, Y., Yu, D., Weng, F., Yuan, Z., Luo, P., Liu, W.,
   & Wang, X. (2022). ByteTrack: Multi-Object Tracking by Associating Every
   Detection Box. *ECCV 2022.* arXiv:2110.06864.
5. Aharon, N., Orfaig, R., & Bobrovsky, B.-Z. (2022). BoT-SORT: Robust
   Associations for Multi-Object Tracking. arXiv:2206.14651.
6. Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You Only Look
   Once: Unified, Real-Time Object Detection. *CVPR 2016.*
7. Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D.,
   Dollár, P., & Zitnick, C. L. (2014). Microsoft COCO: Common Objects in
   Context. *ECCV 2014.*
8. Bewley, A., Ge, Z., Ott, L., Ramos, F., & Upcroft, B. (2016). Simple Online
   and Realtime Tracking. *ICIP 2016.*
9. Wojke, N., Bewley, A., & Paulus, D. (2017). Simple Online and Realtime
   Tracking with a Deep Association Metric. *ICIP 2017.*
10. Bradski, G. (2000). The OpenCV Library. *Dr. Dobb's Journal of Software
    Tools.* <https://opencv.org/>
11. Streamlit. *Streamlit Documentation.* <https://docs.streamlit.io/>

---

## 18. Demonstration checklist

Use this sequence for a class presentation or evaluation:

- [ ] Activate the virtual environment and start `streamlit run app.py`.
- [ ] Show the loaded model name, 80 classes, and resolved device (`CPU` or CUDA).
- [ ] Explain the pipeline diagram: capture → YOLO → association → annotation →
      statistics/output.
- [ ] Select `sample_traffic.avi`, set a 300-frame limit, keep ByteTrack.
- [ ] Start; point out boxes, class labels, confidence scores and persistent `#ID`.
- [ ] Pause, confirm the frame counter stops, then resume.
- [ ] Show unique IDs, class breakdown and per-ID first/last/seen frames.
- [ ] Stop or let it complete; show the verified output panel.
- [ ] Download the annotated MP4 and play a few seconds.
- [ ] Start a second short session and note that IDs reset per session.
- [ ] Show `python -m pytest -q` and `python verification.py` results.
- [ ] Discuss limitations: CPU FPS, occlusion/ID switches, COCO-only classes,
      privacy, and no audio.

---

## 19. Summary

The project delivers a modular, testable real-time detection and tracking
application built on YOLO11, ByteTrack/BoT-SORT, OpenCV and Streamlit. It
preserves the source video's geometry and frame rate, exposes meaningful live
statistics without inventing tracker IDs, verifies every output file, and
handles camera/file failures cleanly. The included test suite and verification
runner make the implementation reproducible and suitable for academic
demonstration.
