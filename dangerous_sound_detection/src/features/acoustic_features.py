import numpy as np

EPSILON = 1e-8

ACOUSTIC_FEATURE_NAMES = (
    "rms_dbfs",
    "peak_dbfs",
    "crest_factor",
    "zero_crossing_rate",
    "spectral_centroid_hz",
    "spectral_rolloff_hz",
    "spectral_flatness",
)


def _safe_db(value: float) -> float:
    return float(20.0 * np.log10(max(value, EPSILON)))


def summarize_acoustic_features(audio: np.ndarray, sr: int) -> dict[str, float]:
    """Compute lightweight acoustic descriptors for a single audio window."""
    y = np.asarray(audio, dtype=np.float32)
    if y.size == 0:
        return {name: 0.0 for name in ACOUSTIC_FEATURE_NAMES}

    peak = float(np.max(np.abs(y)))
    rms = float(np.sqrt(np.mean(np.square(y), dtype=np.float64)))
    crest_factor = float(peak / max(rms, EPSILON))

    sign_changes = np.count_nonzero(np.diff(np.signbit(y)))
    zero_crossing_rate = float(sign_changes / max(y.size - 1, 1))

    window = np.hanning(y.size).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(y * window)).astype(np.float32)
    freqs = np.fft.rfftfreq(y.size, d=1.0 / sr).astype(np.float32)
    magnitude_sum = float(np.sum(spectrum))

    if magnitude_sum <= EPSILON:
        spectral_centroid_hz = 0.0
        spectral_rolloff_hz = 0.0
        spectral_flatness = 0.0
    else:
        spectral_centroid_hz = float(np.sum(freqs * spectrum) / magnitude_sum)
        cumulative_energy = np.cumsum(spectrum)
        rolloff_threshold = 0.85 * magnitude_sum
        rolloff_index = int(np.searchsorted(cumulative_energy, rolloff_threshold, side="left"))
        rolloff_index = min(rolloff_index, len(freqs) - 1)
        spectral_rolloff_hz = float(freqs[rolloff_index])
        spectral_flatness = float(
            np.exp(np.mean(np.log(spectrum + EPSILON))) / max(np.mean(spectrum), EPSILON)
        )

    return {
        "rms_dbfs": _safe_db(rms),
        "peak_dbfs": _safe_db(peak),
        "crest_factor": crest_factor,
        "zero_crossing_rate": zero_crossing_rate,
        "spectral_centroid_hz": spectral_centroid_hz,
        "spectral_rolloff_hz": spectral_rolloff_hz,
        "spectral_flatness": spectral_flatness,
    }


def compute_acoustic_feature_vector(audio: np.ndarray, sr: int) -> np.ndarray:
    """Return acoustic descriptors in a fixed feature order."""
    summary = summarize_acoustic_features(audio, sr)
    return np.asarray([summary[name] for name in ACOUSTIC_FEATURE_NAMES], dtype=np.float32)
