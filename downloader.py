import os
import subprocess
import json
from pathlib import Path
from config import DOWNLOADS_DIR, FRAMES_DIR, LONG_FILM_FRAME_FPS

os.makedirs(DOWNLOADS_DIR, exist_ok=True)
os.makedirs(FRAMES_DIR, exist_ok=True)


def download_screener(screener_url: str, password: str, entry_id: str) -> dict:
    """
    Download film from FilmFreeway screener link (Vimeo/Drive/Dropbox).
    Returns: {"path": str, "duration_min": float, "size_mb": float, "success": bool}
    """
    output_path = Path(DOWNLOADS_DIR) / f"{entry_id}.mp4"

    if output_path.exists():
        print(f"  [cache] {entry_id} already downloaded")
        duration = get_video_duration(str(output_path))
        size = output_path.stat().st_size / (1024 * 1024)
        return {"path": str(output_path), "duration_min": duration,
                "size_mb": size, "success": True}

    cmd = [
        "yt-dlp",
        "--video-password", password,
        "-o", str(output_path),
        "--format", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "--quiet",
        "--no-warnings",
        screener_url
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"  [error] Download failed for {entry_id}: {result.stderr[:200]}")
            return {"path": None, "duration_min": 0, "size_mb": 0, "success": False}

        duration = get_video_duration(str(output_path))
        size = output_path.stat().st_size / (1024 * 1024)
        print(f"  [ok] Downloaded {entry_id} — {duration:.1f} min, {size:.0f} MB")
        return {"path": str(output_path), "duration_min": duration,
                "size_mb": size, "success": True}

    except subprocess.TimeoutExpired:
        print(f"  [timeout] Download timed out for {entry_id}")
        return {"path": None, "duration_min": 0, "size_mb": 0, "success": False}
    except Exception as e:
        print(f"  [exception] {entry_id}: {e}")
        return {"path": None, "duration_min": 0, "size_mb": 0, "success": False}


def get_video_duration(video_path: str) -> float:
    """Return video duration in minutes using ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        video_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        data = json.loads(result.stdout)
        duration_sec = float(data["format"]["duration"])
        return duration_sec / 60
    except Exception:
        return 0.0


def extract_keyframes(video_path: str, entry_id: str) -> list[str]:
    """
    Extract 1 keyframe every 10 seconds for long film analysis.
    Used when film > SHORT_FILM_MAX_MIN.
    Returns list of frame paths.
    """
    frame_dir = Path(FRAMES_DIR) / entry_id
    frame_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"fps={LONG_FILM_FRAME_FPS}",
        "-q:v", "2",
        str(frame_dir / "frame_%04d.jpg"),
        "-y", "-loglevel", "quiet"
    ]

    try:
        subprocess.run(cmd, timeout=300, check=True)
        frames = sorted(frame_dir.glob("*.jpg"))
        print(f"  [frames] Extracted {len(frames)} keyframes for {entry_id}")
        return [str(f) for f in frames]
    except Exception as e:
        print(f"  [error] Frame extraction failed: {e}")
        return []


def transcribe_audio(video_path: str, entry_id: str) -> str:
    """
    Use whisper.cpp (already on M4 Pro) to transcribe audio.
    Falls back gracefully if whisper not available.
    """
    transcript_path = Path(DOWNLOADS_DIR) / f"{entry_id}_transcript.txt"

    if transcript_path.exists():
        return transcript_path.read_text()

    # Extract audio first
    audio_path = Path(DOWNLOADS_DIR) / f"{entry_id}_audio.wav"
    audio_cmd = [
        "ffmpeg", "-i", video_path,
        "-ar", "16000", "-ac", "1",
        str(audio_path), "-y", "-loglevel", "quiet"
    ]

    try:
        subprocess.run(audio_cmd, timeout=120, check=True)

        # Try whisper.cpp (path may vary — adjust to your install)
        whisper_cmd = [
            "./whisper.cpp/main",
            "-m", "./whisper.cpp/models/ggml-base.en.bin",
            "-f", str(audio_path),
            "-otxt",
            "-of", str(transcript_path.with_suffix("")),
            "--language", "auto"
        ]
        result = subprocess.run(whisper_cmd, capture_output=True,
                                text=True, timeout=600)
        if transcript_path.exists():
            transcript = transcript_path.read_text()
            print(f"  [whisper] Transcribed {entry_id} ({len(transcript)} chars)")
            return transcript

    except Exception as e:
        print(f"  [whisper] Not available or failed: {e}. Proceeding without transcript.")

    return ""
