import numpy as np
from src.utils.audio import load_audio, preprocess_audio, window_audio
from src.features.yamnet_extractor import YAMNetExtractor
import joblib

def predict_audio(audio_path: str, config: dict, extractor=None, clf=None, le=None) -> dict:
    """Predict events in audio file."""
    extractor = extractor or YAMNetExtractor(config['yamnet_model_url'])
    clf = clf or joblib.load(config['model_save_path'])
    le = le or joblib.load(config['label_encoder_path'])
    y = load_audio(audio_path, config['sample_rate'])
    y = preprocess_audio(y, config['sample_rate'])
    frames = window_audio(y, config['window_length'], config['hop_length'], config['sample_rate'])
    predictions = []
    for i, frame in enumerate(frames):
        emb = extractor.extract(frame, config['sample_rate'])
        feat = emb.mean(axis=0)
        pred_proba = clf.predict_proba([feat])[0]
        class_idx = np.argmax(pred_proba)
        confidence = pred_proba[class_idx]
        label = le.inverse_transform([class_idx])[0]
        if confidence > config['confidence_threshold'] and label in ['gunshot', 'explosion']:
            start_time = i * config['hop_length']
            end_time = start_time + config['window_length']
            predictions.append({
                'label': label,
                'start_time': start_time,
                'end_time': end_time,
                'confidence': float(confidence)
            })
    # Simple post-processing: merge close events
    merged = []
    for pred in predictions:
        if not merged or pred['start_time'] - merged[-1]['end_time'] > 0.5:
            merged.append(pred)
        else:
            merged[-1]['end_time'] = pred['end_time']
            merged[-1]['confidence'] = max(merged[-1]['confidence'], pred['confidence'])
    return {'source': audio_path, 'events': merged}
