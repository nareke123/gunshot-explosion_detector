import numpy as np
import pandas as pd
from src.utils.audio import load_audio, preprocess_audio, window_audio
from src.features.acoustic_features import ACOUSTIC_FEATURE_NAMES, compute_acoustic_feature_vector
from src.features.yamnet_extractor import YAMNetExtractor
from src.utils.io import save_features, save_labels

def extract_features_for_file(
    file_path: str,
    extractor: YAMNetExtractor,
    window_length: float,
    hop_length: float,
    sr: int,
) -> np.ndarray:
    """Extract one feature vector per audio window for a single file."""
    y = load_audio(file_path, sr)
    y = preprocess_audio(y, sr, normalize=False)
    frames = window_audio(y, window_length, hop_length, sr)
    features = []
    for frame in frames:
        emb = extractor.extract(frame, sr)
        spectral_features = emb.mean(axis=0)
        acoustic_features = compute_acoustic_feature_vector(frame, sr)
        features.append(np.concatenate([spectral_features, acoustic_features]).astype(np.float32))
    if features:
        return np.asarray(features, dtype=np.float32)
    feature_dim = getattr(extractor, 'feature_dim', 257) + len(ACOUSTIC_FEATURE_NAMES)
    return np.zeros((1, feature_dim), dtype=np.float32)

def extract_all_features(splits_csv: str, extractor: YAMNetExtractor, config: dict) -> None:
    """Extract one feature vector per audio window for all files in a split."""
    df = pd.read_csv(splits_csv)
    features = []
    labels = []
    for _, row in df.iterrows():
        file_features = extract_features_for_file(
            row['file'],
            extractor,
            config['window_length'],
            config['hop_length'],
            config['sample_rate'],
        )
        features.extend(file_features)
        labels.extend([row['label']] * len(file_features))
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels)
    save_features(features, config['features_path'])
    save_labels(labels, config['labels_path'])
