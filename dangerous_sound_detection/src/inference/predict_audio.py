import numpy as np
from src.utils.audio import load_audio, preprocess_audio, window_audio
from src.features.acoustic_features import compute_acoustic_feature_vector, summarize_acoustic_features
from src.features.yamnet_extractor import YAMNetExtractor
from src.inference.event_gate import estimate_noise_floor_dbfs, passes_event_gate
import joblib
from src.training.torch_model import load_torch_classifier, predict_proba


def _merge_predictions(predictions: list[dict], gap_seconds: float) -> list[dict]:
    merged = []
    for pred in predictions:
        if (
            not merged
            or pred['label'] != merged[-1]['label']
            or pred['start_time'] - merged[-1]['end_time'] > gap_seconds
        ):
            merged.append(pred.copy())
        else:
            merged[-1]['end_time'] = pred['end_time']
            merged[-1]['confidence'] = max(merged[-1]['confidence'], pred['confidence'])
    return merged


def predict_audio(
    audio_path: str,
    config: dict,
    extractor=None,
    model=None,
    le=None,
    device=None,
    confidence_threshold: float | None = None,
    include_windows: bool = False,
) -> dict:
    """Predict sound events in an audio file using a gate + classifier cascade."""
    extractor = extractor or YAMNetExtractor(
        device=config.get('device', 'auto'),
        n_fft=config.get('n_fft', 512),
        hop_length=config.get('stft_hop_length', 160),
        win_length=config.get('win_length', 400),
    )
    if model is None:
        model, device = load_torch_classifier(config['model_save_path'], config.get('device', 'auto'))
    elif device is None:
        device = config.get('device', 'auto')
    le = le or joblib.load(config['label_encoder_path'])
    background_labels = set(config.get('background_labels', ['background', 'normal']))
    alert_labels = set(
        config.get(
            'alert_labels',
            [label for label in config.get('model_class_names', []) if label not in background_labels],
        )
    )
    event_labels = set(
        config.get(
            'event_labels',
            [label for label in le.classes_ if label not in background_labels],
        )
    )
    threshold = float(
        config['confidence_threshold'] if confidence_threshold is None else confidence_threshold
    )
    y = load_audio(audio_path, config['sample_rate'])
    y = preprocess_audio(y, config['sample_rate'], normalize=False)
    frames = window_audio(y, config['window_length'], config['hop_length'], config['sample_rate'])
    acoustic_summaries = [
        summarize_acoustic_features(frame, config['sample_rate'])
        for frame in frames
    ]
    noise_floor_dbfs = estimate_noise_floor_dbfs(
        acoustic_summaries,
        percentile=float(config.get('event_gate_noise_floor_percentile', 20.0)),
    )
    all_events = []
    alert_predictions = []
    window_predictions = []
    merge_gap_seconds = float(config.get('merge_gap_seconds', 0.5))
    for i, (frame, acoustic_summary) in enumerate(zip(frames, acoustic_summaries)):
        start_time = i * config['hop_length']
        end_time = start_time + config['window_length']
        emb = extractor.extract(frame, config['sample_rate'])
        feat = np.concatenate(
            [emb.mean(axis=0), compute_acoustic_feature_vector(frame, config['sample_rate'])]
        ).astype(np.float32)
        pred_proba = predict_proba(model, [feat], device)[0]
        class_idx = np.argmax(pred_proba)
        confidence = float(pred_proba[class_idx])
        label = le.inverse_transform([class_idx])[0]
        gate_passed, gate_details = passes_event_gate(acoustic_summary, config, noise_floor_dbfs)
        if include_windows:
            window_predictions.append({
                'start_time': start_time,
                'end_time': end_time,
                'predicted_label': str(label),
                'confidence': confidence,
                'event_gate': gate_details,
                'acoustic_summary': acoustic_summary,
                'probabilities': {
                    str(class_name): float(prob)
                    for class_name, prob in zip(le.classes_, pred_proba)
                },
            })

        if gate_passed and confidence > threshold and label in event_labels:
            prediction = {
                'label': label,
                'start_time': start_time,
                'end_time': end_time,
                'confidence': confidence
            }
            all_events.append(prediction)
            if label in alert_labels:
                alert_predictions.append(prediction)

    result = {
        'source': audio_path,
        'confidence_threshold': threshold,
        'noise_floor_dbfs': noise_floor_dbfs,
        'events': _merge_predictions(alert_predictions, merge_gap_seconds),
        'activities': _merge_predictions(all_events, merge_gap_seconds),
    }
    if include_windows:
        result['windows'] = window_predictions
    return result
