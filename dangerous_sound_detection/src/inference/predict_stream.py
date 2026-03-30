import subprocess
from collections import deque
import numpy as np
from src.features.acoustic_features import compute_acoustic_feature_vector, summarize_acoustic_features
from src.features.yamnet_extractor import YAMNetExtractor
from src.inference.event_gate import estimate_noise_floor_dbfs, passes_event_gate
import joblib
import logging
from src.utils.logging_utils import setup_logging
from src.training.torch_model import load_torch_classifier, predict_proba

def predict_stream(rtsp_url: str, config: dict) -> None:
    """MVP RTSP stream prediction. Prints alerts to console."""
    setup_logging()
    logger = logging.getLogger(__name__)
    extractor = YAMNetExtractor(
        device=config.get('device', 'auto'),
        n_fft=config.get('n_fft', 512),
        hop_length=config.get('stft_hop_length', 160),
        win_length=config.get('win_length', 400),
    )
    model, device = load_torch_classifier(config['model_save_path'], config.get('device', 'auto'))
    le = joblib.load(config['label_encoder_path'])
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
    # Use ffmpeg to pipe audio
    frame_samples = int(config['window_length'] * config['sample_rate'])
    chunk_size = frame_samples * 2  # int16 mono samples
    cmd = [
        'ffmpeg',
        '-i', rtsp_url,
        '-f', 's16le',
        '-acodec', 'pcm_s16le',
        '-ac', '1',
        '-ar', str(config['sample_rate']),
        '-',
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    logger.info("Starting RTSP stream inference...")
    buffer = b''
    acoustic_history = deque(maxlen=int(config.get('event_gate_history_size', 32)))
    try:
        while True:
            data = proc.stdout.read(chunk_size)
            if not data:
                break
            buffer += data
            if len(buffer) >= chunk_size:
                y = np.frombuffer(buffer[:chunk_size], dtype=np.int16).astype(np.float32) / 32768.0
                buffer = buffer[chunk_size:]
                acoustic_summary = summarize_acoustic_features(y, config['sample_rate'])
                noise_floor_dbfs = estimate_noise_floor_dbfs(
                    list(acoustic_history) or [acoustic_summary],
                    percentile=float(config.get('event_gate_noise_floor_percentile', 20.0)),
                )
                gate_passed, gate_details = passes_event_gate(acoustic_summary, config, noise_floor_dbfs)
                acoustic_history.append(acoustic_summary)
                emb = extractor.extract(y, config['sample_rate'])
                feat = np.concatenate(
                    [emb.mean(axis=0), compute_acoustic_feature_vector(y, config['sample_rate'])]
                ).astype(np.float32)
                pred_proba = predict_proba(model, [feat], device)[0]
                class_idx = np.argmax(pred_proba)
                confidence = pred_proba[class_idx]
                label = le.inverse_transform([class_idx])[0]
                if gate_passed and confidence > config['confidence_threshold'] and label in alert_labels:
                    logger.warning("ALERT: %s detected with confidence %.2f", label, confidence)
                elif gate_passed and confidence > config['confidence_threshold'] and label in event_labels:
                    logger.info(
                        "Event: %s detected with confidence %.2f (%s)",
                        label,
                        confidence,
                        gate_details['reason'],
                    )
    finally:
        proc.terminate()
