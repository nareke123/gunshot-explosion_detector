import subprocess
import numpy as np
from src.features.yamnet_extractor import YAMNetExtractor
import joblib
import logging
from src.utils.logging_utils import setup_logging

def predict_stream(rtsp_url: str, config: dict) -> None:
    """MVP RTSP stream prediction. Prints alerts to console."""
    setup_logging()
    logger = logging.getLogger(__name__)
    extractor = YAMNetExtractor(config['yamnet_model_url'])
    clf = joblib.load(config['model_save_path'])
    le = joblib.load(config['label_encoder_path'])
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
    try:
        while True:
            data = proc.stdout.read(chunk_size)
            if not data:
                break
            buffer += data
            if len(buffer) >= chunk_size:
                y = np.frombuffer(buffer[:chunk_size], dtype=np.int16).astype(np.float32) / 32768.0
                buffer = buffer[chunk_size:]
                emb = extractor.extract(y, config['sample_rate'])
                feat = emb.mean(axis=0)
                pred_proba = clf.predict_proba([feat])[0]
                class_idx = np.argmax(pred_proba)
                confidence = pred_proba[class_idx]
                label = le.inverse_transform([class_idx])[0]
                if confidence > config['confidence_threshold'] and label in ['gunshot', 'explosion']:
                    logger.warning("ALERT: %s detected with confidence %.2f", label, confidence)
    finally:
        proc.terminate()
