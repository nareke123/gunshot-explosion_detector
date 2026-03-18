# Dangerous Sound Detection MVP

## Overview

This is a Minimum Viable Product (MVP) for detecting dangerous acoustic events from audio/video streams. It focuses on identifying gunshots, explosions, and normal sounds using YAMNet for feature extraction and a simple classifier.

**Important Note:** This is an audio-event detection MVP, not a fully production-validated safety system. It should not be used for critical safety applications without further validation and testing.

## Architecture

- **Feature Extraction:** YAMNet (pretrained from TensorFlow Hub) extracts embeddings from audio windows.
- **Classifier:** LogisticRegression (or optional XGBoost) trained on aggregated embeddings.
- **Inference:** Processes audio in sliding windows, applies post-processing for event detection.
- **Data Pipeline:** Supports audio/video files, extracts audio, preprocesses, and splits into train/val/test.

## Dataset Format

Place your data in `data/raw/` with subfolders:
- `gunshot/`
- `explosion/`
- `normal/`

Supported formats: .wav, .mp3, .mp4, .avi, .mov, .mkv (audio extracted from videos).

## Installation

1. Clone or download the project.
2. Create and activate a virtual environment with Python 3.13.
3. Install dependencies: `pip install -r requirements.txt`
4. Ensure FFmpeg is installed and available in `PATH` for video/RTSP audio extraction.

The project now targets Python 3.13 with TensorFlow `2.20+` and `tensorflow-hub` `0.16.1+`.
On the first YAMNet run, TensorFlow Hub may download the model artifact.

## Usage

### Prepare Dataset
```bash
python -m src.cli prepare-dataset --config configs/default.yaml
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

Edit `configs/default.yaml` for paths, thresholds, etc.

## Limitations

- RTSP streaming is basic and may not handle all cases robustly.
- Classifier is a simple baseline; performance may vary.
- No real-time optimization for production.
- Assumes local CPU/GPU setup.

## Next Improvements

- Add more sophisticated models (e.g., CNN on spectrograms).
- Implement real-time streaming with buffering.
- Add more classes or multi-label detection.
- Integrate with APIs or web interfaces.
- Extensive testing and validation.

## License

MIT License.
