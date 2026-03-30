import os
from pathlib import Path
import subprocess


def resolve_ffmpeg_cmd() -> str:
    """Resolve the ffmpeg executable from PATH, env, or WinGet install folders."""
    env_path = os.environ.get("FFMPEG_BINARY")
    if env_path:
        return env_path

    winget_root = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
    if winget_root.exists():
        matches = sorted(winget_root.glob("Gyan.FFmpeg_*/*/bin/ffmpeg.exe"))
        if matches:
            return str(matches[-1])

    return "ffmpeg"


def extract_audio_from_video(video_path: str, audio_path: str) -> None:
    """Extract audio from video file using ffmpeg."""
    ffmpeg_cmd = resolve_ffmpeg_cmd()
    try:
        subprocess.run(
            [
                ffmpeg_cmd,
                "-i",
                video_path,
                "-ac",
                "1",
                "-ar",
                "16000",
                audio_path,
                "-y",
            ],
            check=True,
            capture_output=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "FFmpeg executable was not found. Install FFmpeg and add it to PATH to process video files."
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(f"FFmpeg failed to extract audio: {stderr}") from exc
