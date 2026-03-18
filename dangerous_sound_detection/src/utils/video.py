import ffmpeg

def extract_audio_from_video(video_path: str, audio_path: str) -> None:
    """Extract audio from video file using ffmpeg."""
    ffmpeg.input(video_path).output(audio_path, ac=1, ar=16000).run(
        overwrite_output=True,
        capture_stdout=True,
        capture_stderr=True,
    )
