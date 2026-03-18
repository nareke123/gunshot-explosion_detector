import numpy as np
import pandas as pd
from src.utils.audio import load_audio, preprocess_audio, window_audio
from src.features.yamnet_extractor import YAMNetExtractor
from src.utils.io import save_features, save_labels

def extract_features_for_file(file_path: str, extractor: YAMNetExtractor, window_length: float, hop_length: float, sr: int) -> np.ndarray:
    """Extract features for a single file."""
    y = load_audio(file_path, sr)
    y = preprocess_audio(y, sr)
    frames = window_audio(y, window_length, hop_length, sr)
    features = []
    for frame in frames:
        emb = extractor.extract(frame, sr)
        features.append(emb.mean(axis=0))  # mean pool per frame
    if features:
        return np.mean(features, axis=0).astype(np.float32)  # mean pool across frames
    return np.zeros(1024, dtype=np.float32)

def extract_all_features(splits_csv: str, extractor: YAMNetExtractor, config: dict) -> None:
    """Extract features for all files in splits."""
    df = pd.read_csv(splits_csv)
    features = []
    labels = []
    for _, row in df.iterrows():
        feat = extract_features_for_file(row['file'], extractor, config['window_length'], config['hop_length'], config['sample_rate'])
        features.append(feat)
        labels.append(row['label'])
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels)
    save_features(features, config['features_path'])
    save_labels(labels, config['labels_path'])
