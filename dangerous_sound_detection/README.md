# Dangerous Sound Event Detection MVP

## Overview

This is a Minimum Viable Product (MVP) for detecting and classifying noisy sound events from audio/video streams. It uses a two-stage pipeline:

- an acoustic event gate that reacts to loud or transient windows relative to the local noise floor
- a PyTorch classifier that labels the event type, for example `background`, `benign_impact`, `loud_children`, `vocal_distress`, or `sharp_impulse`

**Important Note:** This is an audio-event detection MVP, not a fully production-validated safety system. It should not be used for critical safety applications without further validation and testing.

## Architecture

- **Feature Extraction:** PyTorch computes log-spectrogram features from audio windows and augments them with acoustic descriptors such as RMS, peak, crest factor, spectral centroid, and rolloff.
- **Event Gate:** A lightweight rule-based gate reacts only to loud or transient windows, which helps ignore steady classroom background.
- **Classifier:** A small PyTorch MLP is trained on per-window features and predicts the type of noisy event.
- **Inference:** Processes audio in sliding windows, tracks noise floor, and returns both general activities and higher-priority alerts.
- **Data Pipeline:** Supports audio/video files, extracts audio, preserves relative loudness, and splits data into train/val/test.

## Dataset Format

Place your data in `data/raw/` with subfolders:
- `background/`
- `benign_impact/`
- `loud_children/`
- `vocal_distress/`
- `sharp_impulse/`

The default config also maps legacy folders such as `normal`, `gunshot`, and `explosion` into the newer event taxonomy via `label_groups`.

Supported formats: .wav, .mp3, .mp4, .avi, .mov, .mkv (audio extracted from videos).

## Installation

1. Clone or download the project.
2. Create and activate a virtual environment with Python 3.13.
3. Install dependencies: `pip install -r requirements.txt`
4. Ensure FFmpeg is installed and available in `PATH` for video/RTSP audio extraction.

The project now targets Python 3.13 with PyTorch `2.5+`.
If you want GPU training, install a CUDA-enabled PyTorch build for your system; otherwise the code will fall back to CPU automatically.

## Usage

### Prepare Dataset
```bash
python -m src.cli prepare-dataset --config configs/default.yaml
```

### Import UrbanSound8K as Gunshot/Normal
```bash
python -m src.cli import-urbansound8k --archive data/archive --raw-dir data/raw
```

### Make Splits
```bash
python -m src.cli make-splits --config configs/default.yaml
```

### Extract Features
```bash
python -m src.cli extract-features --config configs/default.yaml
```

### Train Model
```bash
python -m src.cli train --config configs/default.yaml
```

### Evaluate
```bash
python -m src.cli evaluate --config configs/default.yaml
```

### Predict on Audio
```bash
python -m src.cli predict-audio --input path/to/audio.wav --output results.json --config configs/default.yaml
```

### Predict on Video
```bash
python -m src.cli predict-video --input path/to/video.mp4 --output results.json --config configs/default.yaml
```

### Predict on Stream (MVP)
```bash
python -m src.cli predict-stream --rtsp rtsp://example.com/stream --config configs/default.yaml
```

## Configuration

Edit `configs/default.yaml` for paths, thresholds, labels, and event gate behavior.

Important settings:

- `background_labels`: labels treated as regular ambient sound
- `event_labels`: noisy foreground event labels
- `alert_labels`: labels that should raise alerts
- `event_gate_*`: loudness and transient thresholds for the first-stage detector

## Limitations

- RTSP streaming is basic and may not handle all cases robustly.
- Classifier is a simple baseline; performance will depend heavily on hard negatives such as desk movement, chairs, doors, and loud children.
- No real-time optimization for production.
- Assumes local CPU/GPU setup.

## Next Improvements

- Add more sophisticated models (e.g., CNN or CRNN on spectrograms).
- Implement real-time streaming with buffering.
- Add richer event taxonomies or multi-label detection.
- Integrate with APIs or web interfaces.
- Extensive testing and validation.

## License

MIT License.
