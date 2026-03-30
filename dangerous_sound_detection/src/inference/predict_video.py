from pathlib import Path
import uuid
from src.utils.video import extract_audio_from_video
from src.inference.predict_audio import predict_audio

def predict_video(
    video_path: str,
    config: dict,
    confidence_threshold: float | None = None,
    include_windows: bool = False,
) -> dict:
    """Predict events in video file by extracting audio."""
    tmp_dir = Path('.tmp')
    tmp_dir.mkdir(parents=True, exist_ok=True)
    audio_path = tmp_dir / f'extracted_audio_{uuid.uuid4().hex}.wav'
    try:
        extract_audio_from_video(video_path, str(audio_path))
        return predict_audio(
            str(audio_path),
            config,
            confidence_threshold=confidence_threshold,
            include_windows=include_windows,
        )
    finally:
        audio_path.unlink(missing_ok=True)
