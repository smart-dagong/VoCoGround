# VoCoGround: Real-Time Camera + Voice Query + API Grounding

This project implements an end-to-end interactive workflow:

- Choose an input source at startup: live camera or local image upload
- Start and stop voice input via on-screen buttons
- When voice input stops, send the current image and text query to the model
- Send transcribed speech + image to a vision API
- Receive grounding boxes and overlay results in the app window

The core goal is to let users watch a live view, describe targets by voice, and quickly see visual grounding results.

---

## 1. Features Overview

### Integrated Main Flow

Main script: `realtime_voice_camera_grounding.py`

Interactive capabilities:

- Startup mode: `camera` (live camera) / `image` (uploaded image)
- In-window buttons: start recording / stop recording
- Voice text confirmation: show ASR text first, allow manual edit (Chinese input supported), then confirm upload
- Inference trigger on stop:
  - `camera` mode uses the frame captured at stop-click
  - `image` mode uses the selected image
- Speech recognition: Whisper converts recording to text
- Visual grounding: Qwen-VL returns target boxes
- Multi-target support: one query can return multiple boxes
- Real-time visualization: overlays grounding results on the main window

### Other Scripts (Modular Tools)

- `camera_capture.py`: standalone photo capture utility
- `whisper_model.py`: standalone recording + transcription utility
- `qwen_vl_ref_grounding.py`: standalone image-text grounding and box drawing utility

---

## 2. Project Structure

```text
apikey_config.json
camera_capture.py
qwen_vl_ref_grounding.py
realtime_voice_camera_grounding.py
whisper_model.py
photos/
```

---

## 3. Requirements

- Python 3.9+
- Available microphone
- ffmpeg (required by Whisper dependencies)

Notes:

- `camera` mode requires a camera device
- `image` mode does not require a camera

---

## 4. Install Dependencies

```bash
pip install -U opencv-python sounddevice numpy openai-whisper openai Pillow
```

If you want drag-and-drop image selection in the startup dialog, install:

```bash
pip install tkinterdnd2
```

Package roles:

- `opencv-python`: camera capture and UI rendering
- `sounddevice` + `numpy`: microphone recording
- `openai-whisper`: speech-to-text
- `openai`: OpenAI-compatible API client (for Qwen-VL)
- `Pillow`: image drawing/annotation support

---

## 5. API Configuration

By default, the app reads `apikey_config.json`:

```json
{
  "api_key": "your API key",
  "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
  "model": "qwen3.5-vl-plus"
}
```

You can also use environment variables (default key name: `DASHSCOPE_API_KEY`).

---

## 6. Quick Start

Run the integrated script:

```bash
python realtime_voice_camera_grounding.py
```

Choose startup source:

```bash
# Recommended: launch and choose camera/image in popup
python realtime_voice_camera_grounding.py

# Force camera mode from CLI
python realtime_voice_camera_grounding.py --startup-mode camera

# Image mode with direct path (optional)
python realtime_voice_camera_grounding.py --startup-mode image --input-image photos/demo.jpg
```

Optional argument example:

```bash
python realtime_voice_camera_grounding.py --camera-index 0 --whisper-model turbo --device auto
```

Common arguments:

```text
--camera-index      Camera index, default 0
--width             Capture width, default 1280
--height            Capture height, default 720
--sample-rate       Audio sample rate, default 16000
--startup-mode      Startup source: auto/camera/image, default auto
--input-image       Image path for image mode
--whisper-model     tiny/base/small/medium/large/turbo
--device            auto/cpu/cuda
--source-language   Input language (e.g. zh/en), default auto
--api-config        API config file path, default apikey_config.json
--model             Vision model name, default qwen3.5-vl-plus
--max-boxes         Max boxes to keep, default 20 (0 = unlimited)
--result-json       Output JSON path for boxes, default latest_boxes.json (empty disables saving)
```

Grounding result JSON example:

```json
{
  "query": "highlight people in red on the left",
  "count": 3,
  "model": "qwen3.5-vl-plus",
  "boxes": [
    {"bbox": [120, 88, 260, 420], "label": "person_red_left1"},
    {"bbox": [300, 96, 438, 430], "label": "person_red_center1"}
  ]
}
```

Note: labels are auto-numbered (same class/label becomes `car1`, `car2`, `car3`, ... in appearance order).

---

## 7. Interaction Flow

### Camera Mode (Live Camera)

1. Launch the app and open the live camera window.
2. Click Start Recording.
3. Speak your query, for example: highlight the person in red on the left.
4. Click Stop Recording.
5. The app transcribes speech first and shows `Editable Text`.
6. Use `Input Text` to open the text editor and modify the query.
7. Chinese input is supported in the editor.
8. Click `Confirm Upload` to submit.
9. The app runs image + text grounding and returns boxes.
10. Results are overlaid on the live view.
11. Click `Clear All` to reset current text.
12. Click `Change Image` to switch image during runtime (no restart required).
13. Click `Exit` (or press `q`) to quit.

Extra: you can skip recording, enter text via `Input Text`, and click `Confirm Upload` directly. New voice input appends to existing text rather than replacing it.

### Image Mode (Uploaded Image + Voice Query)

1. Launch the app and choose `Image` in the startup dialog.
2. Drag an image into the drop area (or click Browse).
3. Click Start to load and display the image.
4. Click Start Recording.
5. Speak your query, for example: highlight the red cup on the left.
6. Click Stop Recording.
7. The app shows transcribed text first; you can edit it via `Input Text`.
8. Chinese input is supported in the editor.
9. The vision API is called only after you click `Confirm Upload`.
10. The app performs text + image grounding and overlays results.
11. Click `Clear All` to reset and rewrite the query.
12. Click `Change Image` to switch to another image and continue.
13. Repeat as needed; click `Exit` (or press `q`) to quit.

Extra: in image mode, you can also skip recording and submit text directly.

Extra: if `tkinterdnd2` is not installed, drag-and-drop is disabled and the dialog will show a hint. Browse still works.

---

## 8. Debugging Tips

If something goes wrong, check first:

- Microphone permission
- Whether the camera is occupied by another program
- Whether API key, base_url, and model are correct
- Whether all dependencies are installed in the active Python environment

### Common Environment Fixes

1. `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x`

This is a compatibility issue between `torch/openai-whisper` and `numpy 2.x`. Run:

```bash
pip install "numpy<2" --upgrade
```

2. `FileNotFoundError` on `SSL_CERT_FILE`

This means `SSL_CERT_FILE` points to a missing certificate file. In PowerShell, run:

```powershell
Remove-Item Env:SSL_CERT_FILE -ErrorAction SilentlyContinue
python realtime_voice_camera_grounding.py
```

The script also includes a fallback: if an invalid cert path is detected, it clears it and tries `certifi` automatically.

You can also test each module separately:

- `python camera_capture.py`
- `python whisper_model.py`
- `python qwen_vl_ref_grounding.py --image photos/xxx.jpg --query "highlight target"`

---

## 9. Notes

This project is for learning and demonstrating a multimodal interaction loop:
voice query + real-time visual understanding + result visualization.

Potential extensions:

- Continuous tracking for same-category targets
- Multi-class target filtering
- Wake-word plus automatic start/stop recording
- Themed UI and richer status panel
