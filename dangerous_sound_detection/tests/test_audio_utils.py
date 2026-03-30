from pathlib import Path
import uuid
import numpy as np
import soundfile as sf
from src.utils.audio import load_audio, normalize_audio, preprocess_audio, trim_silence, window_audio

def test_load_audio():
    tmp_dir = Path('.tmp')
    tmp_dir.mkdir(parents=True, exist_ok=True)
    audio_path = tmp_dir / f'test_{uuid.uuid4().hex}.wav'
    y = np.stack([np.random.randn(8000), np.random.randn(8000)], axis=1).astype(np.float32)
    sf.write(audio_path, y, 8000)
    loaded = load_audio(str(audio_path), sr=16000)
    assert loaded.ndim == 1
    assert loaded.dtype == np.float32
    assert len(loaded) > 0

def test_preprocess_audio():
    y = np.random.randn(16000).astype(np.float32)
    y_proc = preprocess_audio(y, 16000)
    assert y_proc.ndim == 1
    assert np.max(np.abs(y_proc)) <= 1.0


def test_preprocess_audio_can_preserve_relative_loudness():
    y = np.concatenate([np.zeros(1000, dtype=np.float32), np.full(4000, 0.2, dtype=np.float32)])
    y_proc = preprocess_audio(y, 16000, normalize=False)
    assert y_proc.ndim == 1
    assert np.isclose(np.max(np.abs(y_proc)), 0.2)


def test_trim_and_normalize_helpers():
    y = np.concatenate([np.zeros(32, dtype=np.float32), np.array([0.1, -0.2, 0.3], dtype=np.float32)])
    trimmed = trim_silence(y)
    normalized = normalize_audio(trimmed)
    assert trimmed.shape[0] == 3
    assert np.isclose(np.max(np.abs(normalized)), 1.0)

def test_window_audio_pads_short_signal():
    y = np.ones(4000, dtype=np.float32)
    frames = window_audio(y, window_length=0.96, hop_length=0.48, sr=16000)
    assert frames.shape == (1, 15360)
