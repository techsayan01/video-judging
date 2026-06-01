import os
import re
import subprocess
import json
from pathlib import Path
from config import DOWNLOADS_DIR, FRAMES_DIR, LONG_FILM_FRAME_FPS

# Optional residential proxy for YouTube downloads from cloud IPs.
# Set YT_PROXY=http://user:pass@host:port in env / Secret Manager.
# Placeholder value "none" (used when no proxy is configured) is treated as empty.
_raw_proxy = os.getenv("YT_PROXY", "").strip()
_YT_PROXY  = "" if _raw_proxy.lower() in ("", "none", "null", "false") else _raw_proxy

def _yt_base_flags() -> list[str]:
    """Common yt-dlp flags shared across all download calls."""
    flags = [
        "--extractor-retries", "3",
        "--retries", "5",
        "--fragment-retries", "5",
        "--add-header", "User-Agent:Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    ]
    if _YT_PROXY:
        flags += ["--proxy", _YT_PROXY]
    return flags


def _safe_entry_id(entry_id: str) -> str:
    """Allow only hex/alphanumeric chars to prevent path traversal."""
    safe = re.sub(r"[^a-zA-Z0-9_-]", "", entry_id)[:128]
    if not safe:
        raise ValueError(f"Invalid entry_id: {entry_id!r}")
    return safe


def stream_to_gcs(screener_url: str, password: str, bucket: str, blob_name: str) -> tuple[str, float]:
    """
    Pipe yt-dlp stdout directly into a GCS object — no local disk write.
    Returns (gs_uri, duration_min).
    Raises on failure.
    """
    from google.cloud import storage as gcs_lib

    cmd = [
        "yt-dlp",
        *_yt_base_flags(),
        "-o", "-",
        "--format", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "--quiet", "--no-warnings",
    ]
    if password:
        cmd += ["--video-password", password]
    cmd.append(screener_url)

    client = gcs_lib.Client()
    blob   = client.bucket(bucket).blob(blob_name)

    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
        blob.upload_from_file(proc.stdout, content_type="video/mp4")
        proc.wait()
        if proc.returncode not in (0, None):
            err = proc.stderr.read().decode(errors="ignore")[:300]
            raise RuntimeError(f"yt-dlp failed: {err}")

    gs_uri = f"gs://{bucket}/{blob_name}"
    print(f"  [gcs] Streamed to {gs_uri}")

    # Get duration via ffprobe on the GCS-signed URL
    signed = blob.generate_signed_url(expiration=300, method="GET",
                                       version="v4")
    duration_min = get_video_duration(signed)
    return gs_uri, duration_min


def delete_from_gcs(bucket: str, blob_name: str):
    """Delete a GCS object. Silent if already gone."""
    try:
        from google.cloud import storage as gcs_lib
        gcs_lib.Client().bucket(bucket).blob(blob_name).delete()
        print(f"  [gcs] Deleted gs://{bucket}/{blob_name}")
    except Exception as e:
        print(f"  [gcs] Delete failed (non-fatal): {e}")


def generate_upload_url(bucket: str, blob_name: str, content_type: str = "video/mp4",
                        expiration_s: int = 1800) -> str:
    """Return a v4 signed URL allowing a browser PUT directly to GCS.
    Lets clients upload large files without passing through Cloud Run's 32 MB limit.
    """
    from google.cloud import storage as gcs_lib
    import google.auth
    from google.auth.transport import requests as gauth_requests

    creds, _ = google.auth.default()
    # Refresh so we have an access token + the SA email for IAM-based signing
    creds.refresh(gauth_requests.Request())

    blob = gcs_lib.Client().bucket(bucket).blob(blob_name)
    return blob.generate_signed_url(
        version="v4",
        expiration=expiration_s,
        method="PUT",
        content_type=content_type,
        service_account_email=getattr(creds, "service_account_email", None),
        access_token=creds.token,
    )


def download_from_gcs(bucket: str, blob_name: str, dest_path: str) -> dict:
    """Download a GCS object to a local path. Returns duration + size info."""
    from google.cloud import storage as gcs_lib
    blob = gcs_lib.Client().bucket(bucket).blob(blob_name)
    if not blob.exists():
        return {"path": None, "duration_min": 0, "size_mb": 0, "success": False,
                "error": "Uploaded file not found in storage"}
    blob.download_to_filename(dest_path)
    duration = get_video_duration(dest_path)
    size = Path(dest_path).stat().st_size / (1024 * 1024)
    print(f"  [gcs] Downloaded gs://{bucket}/{blob_name} — {duration:.1f} min, {size:.0f} MB")
    return {"path": dest_path, "duration_min": duration, "size_mb": size, "success": True}

os.makedirs(DOWNLOADS_DIR, exist_ok=True)
os.makedirs(FRAMES_DIR, exist_ok=True)


def download_screener(screener_url: str, password: str, entry_id: str) -> dict:
    """
    Download film from FilmFreeway screener link (Vimeo/Drive/Dropbox).
    Returns: {"path": str, "duration_min": float, "size_mb": float, "success": bool}
    """
    entry_id    = _safe_entry_id(entry_id)
    output_path = Path(DOWNLOADS_DIR) / f"{entry_id}.mp4"
    # Resolve and confirm the path stays inside DOWNLOADS_DIR (path traversal guard)
    output_path = output_path.resolve()
    if not str(output_path).startswith(str(Path(DOWNLOADS_DIR).resolve())):
        raise ValueError(f"Path traversal detected for entry_id: {entry_id}")

    if output_path.exists():
        print(f"  [cache] {entry_id} already downloaded")
        duration = get_video_duration(str(output_path))
        size = output_path.stat().st_size / (1024 * 1024)
        return {"path": str(output_path), "duration_min": duration,
                "size_mb": size, "success": True}

    cmd = [
        "yt-dlp",
        *_yt_base_flags(),
        "--video-password", password,
        "-o", str(output_path),
        "--format", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "--newline",          # one progress line per update (machine-readable)
        "--no-warnings",
        screener_url
    ]

    _DOWNLOAD_TIMEOUT = 480   # 8 min max — Vimeo feature films can be 2–4 GB

    try:
        stderr_lines = []
        last_progress_pct = None

        with subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,   # merge stderr into stdout so we see all output
            text=True,
            bufsize=1,
        ) as proc:
            import select, time as _time
            deadline = _time.monotonic() + _DOWNLOAD_TIMEOUT
            for line in proc.stdout:
                if _time.monotonic() > deadline:
                    proc.kill()
                    print(f"  [timeout] Download timed out for {entry_id}")
                    return {"path": None, "duration_min": 0, "size_mb": 0,
                            "success": False, "error": "Download timed out (>8 min)"}
                line = line.rstrip()
                if line:
                    stderr_lines.append(line[-300:])
                    # Print yt-dlp progress so it appears in Cloud Run logs
                    print(f"  [dl] {line}", flush=True)
            proc.wait()

        if proc.returncode != 0:
            stderr_snippet = "\n".join(stderr_lines[-5:]) or "no output"
            print(f"  [error] Download failed for {entry_id}: {stderr_snippet}")
            return {"path": None, "duration_min": 0, "size_mb": 0, "success": False,
                    "error": stderr_snippet}

        if not output_path.exists():
            return {"path": None, "duration_min": 0, "size_mb": 0, "success": False,
                    "error": "yt-dlp exited 0 but output file missing"}

        duration = get_video_duration(str(output_path))
        size = output_path.stat().st_size / (1024 * 1024)
        print(f"  [ok] Downloaded {entry_id} — {duration:.1f} min, {size:.0f} MB")
        return {"path": str(output_path), "duration_min": duration,
                "size_mb": size, "success": True}

    except Exception as e:
        print(f"  [exception] {entry_id}: {e}")
        return {"path": None, "duration_min": 0, "size_mb": 0, "success": False,
                "error": str(e)[:300]}


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
    entry_id  = _safe_entry_id(entry_id)
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
