# Violence Detection System

Real-time violence detection from a webcam or an image, built on OpenAI's **CLIP** model. It runs a live dashboard, sounds an alarm when violence is detected, and saves a snapshot and a video clip of every incident to a searchable log.

No custom training is needed. CLIP compares each frame against short text descriptions such as *"fight on a street"* or *"fire in office"*, so you can change what it detects by editing a text file.

## Features

- **Zero-shot detection.** Detects fights, street violence, fire and car crashes (and more) by matching frames to text labels. Edit `settings.yaml` to add your own.
- **Live web dashboard.** A Flask app streams the camera feed, shows the current status, and lists past incidents.
- **Smart incident recording.**
  - Keeps a 3-second pre-roll buffer, so the clip starts *before* the violence was detected.
  - Keeps recording for 20 seconds after the last detection, so short bursts merge into one clip.
  - Saves a snapshot (`.jpg`) and a clip (`.mp4`) for each incident.
- **Incident history.** Every incident (time, label, confidence, snapshot, clip) is stored in a local SQLite database.
- **Audio alarm.** A looping alarm plays while violence is detected and stops 2 seconds after it ends.
- **Reusable `Model` class.** Use it from your own code (see below).

## How it works

```
Camera frame ──► CLIP image encoder ─┐
                                     ├─► cosine similarity ─► best label
Text labels ───► CLIP text encoder ──┘        │
(settings.yaml)                               ├─ score ≥ threshold ─► that label
                                              └─ score <  threshold ─► "Unknown"
```

If the best label is one of the violent labels, the app starts recording and raises the alert.

## Tech stack

Python · PyTorch · OpenAI CLIP (ViT-B/32) · OpenCV · Flask · SQLite · pygame

## Project structure

| File | Purpose |
|---|---|
| `app.py` | Flask dashboard: live feed, incident recording, history API |
| `index.html` | Dashboard front end |
| `live_detect.py` | Desktop mode: webcam window with on-screen label and audio alarm |
| `model.py` | `Model` class wrapping CLIP and label matching |
| `run.py` | Classify a single image from the command line |
| `database.py` | SQLite helpers for the incident log |
| `settings.yaml` | Model name, device, threshold and labels |
| `tutorial.ipynb` | Notebook with examples |
| `warning-alarm.wav` | Alarm sound |

## Getting started

**Requirements:** Python 3.9+ and a webcam. A GPU is optional (see Configuration).

```bash
git clone <your-repo-url>
cd violence-detection-v0.2
pip install -r requirements.txt
```

The first run downloads the CLIP weights (about 350 MB).

### Run the web dashboard

```bash
python app.py
```

Your browser opens at <http://127.0.0.1:5000>. Detected incidents are saved in `detections/` and logged in `detections.db`.

### Run the desktop window with alarm

```bash
python live_detect.py
```

Press `q` to quit.

### Classify one image

```bash
python run.py --image-path path/to/image.jpg
```

### Use it in your own code

```python
import cv2
from model import Model

model = Model()
image = cv2.imread("your_image.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # the model expects RGB

result = model.predict(image=image)
print(result["label"], result["confidence"])
```

## Configuration

Everything is set in `settings.yaml`:

```yaml
model-settings:
  prediction-threshold: 0.23   # minimum cosine similarity to accept a label
  model-name: 'ViT-B/32'
  device: 'cpu'                # use 'cuda' if you have an NVIDIA GPU
label-settings:
  labels:
    - 'fight on a street'
    - 'fire in office'
    # add your own descriptions here
  default-label: 'Unknown'
```

Which labels count as violence is set in `VIOLENT_LABELS` in `app.py` (dashboard) and `live_detect.py` (desktop mode). If you add a new label, add it there too.

## Limitations

- Each frame is classified on its own, so there is no motion or temporal analysis. A fight is judged by how a single frame looks.
- Accuracy depends on the labels and the threshold. Tune `prediction-threshold` for your camera and scene.
- It is a prototype, not a certified safety system. Do not rely on it alone.

## Roadmap

- Batch processing for faster inference
- Return several top predictions per frame
- Per-scene threshold tuning
- Temporal smoothing across frames to reduce false alarms

## License

Add a license file before sharing the project publicly.
