from pathlib import Path
import tempfile
from src.utils.video import extract_audio_from_video
from src.inference.predict_audio import predict_audio

def predict_video(video_path: str, config: dict) -> dict:
    """Predict events in video file by extracting audio."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        audio_path = Path(tmp_dir) / 'extracted_audio.wav'
        extract_audio_from_video(video_path, str(audio_path))
        return predict_audio(str(audio_path), config)
