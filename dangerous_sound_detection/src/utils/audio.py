import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def load_audio(file_path: str, sr: int = 16000) -> np.ndarray:
    """Load audio file and resample to specified rate."""
    y, original_sr = sf.read(file_path, dtype='float32')
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if original_sr != sr:
        gcd = np.gcd(original_sr, sr)
        y = resample_poly(y, up=sr // gcd, down=original_sr // gcd)
    return np.asarray(y, dtype=np.float32)


def normalize_audio(y: np.ndarray) -> np.ndarray:
    """Scale a waveform to [-1, 1] while preserving zeros."""
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return y
    max_abs = np.max(np.abs(y))
    if max_abs > 0:
        y = y / max_abs
    return np.asarray(y, dtype=np.float32)


def trim_silence(y: np.ndarray, silence_threshold_db: float = -40.0) -> np.ndarray:
    """Trim leading and trailing silence using a threshold relative to the clip peak."""
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return y

    peak = float(np.max(np.abs(y)))
    if peak <= 0.0:
        return y

    silence_threshold = peak * (10 ** (silence_threshold_db / 20))
    non_silent = np.flatnonzero(np.abs(y) > silence_threshold)
    if non_silent.size > 0:
        y = y[non_silent[0]:non_silent[-1] + 1]
    return np.asarray(y, dtype=np.float32)

def preprocess_audio(
    y: np.ndarray,
    sr: int,
    normalize: bool = True,
    silence_threshold_db: float = -40.0,
) -> np.ndarray:
    """Trim silence and optionally normalize amplitude."""
    del sr
    y = trim_silence(y, silence_threshold_db=silence_threshold_db)
    if normalize:
        y = normalize_audio(y)
    return np.asarray(y, dtype=np.float32)


def window_audio(y: np.ndarray, window_length: float, hop_length: float, sr: int) -> np.ndarray:
    """Window audio into frames."""
    frame_length = int(window_length * sr)
    hop_length_samples = int(hop_length * sr)
    if frame_length <= 0 or hop_length_samples <= 0:
        raise ValueError("window_length and hop_length must be positive.")
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return np.zeros((1, frame_length), dtype=np.float32)
    if y.size < frame_length:
        y = np.pad(y, (0, frame_length - y.size))
        return y[np.newaxis, :]

    n_frames = 1 + int(np.ceil((y.size - frame_length) / hop_length_samples))
    total_length = frame_length + (n_frames - 1) * hop_length_samples
    if total_length > y.size:
        y = np.pad(y, (0, total_length - y.size))

    frames = [
        y[start:start + frame_length]
        for start in range(0, total_length - frame_length + 1, hop_length_samples)
    ]
    return np.asarray(frames, dtype=np.float32)
