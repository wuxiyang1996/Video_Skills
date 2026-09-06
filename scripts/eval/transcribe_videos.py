"""Transcribe benchmark videos with timestamps (OpenAI whisper-1) into a per-video JSON cache.

Video-Holmes ships no subtitles and the L1 catalog carries no dialogue at all
(`dialogue_spans` is empty on every clip), while the human segment rows the
questions were written from quote what people say.  This extracts the audio
track with PyAV (no ffmpeg on the login nodes), sends it to whisper-1 with
segment timestamps, and caches `{video_id, language, duration_s, segments:
[{start_s, end_s, text}]}` under `<out-dir>/<video_id>.json`.  Files over the
API size limit are split into fixed-length chunks whose timestamps are offset
back into video time.  Nothing here is question-conditioned.
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import json
import sys
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

SAMPLE_RATE = 16000
MAX_UPLOAD_BYTES = 24 * 1024 * 1024
CHUNK_S = 600.0


def load_openai_api_key(keys_py: Path) -> str:
    spec = importlib.util.spec_from_file_location("keys", keys_py)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    key = getattr(module, "OPENAI_API_KEY", None)
    if not key:
        raise RuntimeError("OPENAI_API_KEY missing in keys.py")
    return key


def extract_pcm16(video_path: Path) -> bytes:
    """Mono 16 kHz PCM16 of the video's first audio stream (empty if none)."""
    import av

    with av.open(str(video_path)) as container:
        streams = [s for s in container.streams if s.type == "audio"]
        if not streams:
            return b""
        resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
        out = bytearray()
        for frame in container.decode(streams[0]):
            for r in resampler.resample(frame):
                out.extend(r.to_ndarray().tobytes())
        return bytes(out)


def wav_bytes(pcm: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)
    return buf.getvalue()


def chunk_pcm(pcm: bytes, chunk_s: float = CHUNK_S) -> list[tuple[float, bytes]]:
    """(offset_s, pcm) chunks no longer than chunk_s, so each upload stays under the size limit."""
    step = int(chunk_s * SAMPLE_RATE) * 2
    return [(i / (SAMPLE_RATE * 2), pcm[i:i + step]) for i in range(0, len(pcm), step)] or [(0.0, b"")]


def merge_segments(chunks: list[tuple[float, dict[str, Any]]]) -> list[dict[str, Any]]:
    """Offset each chunk's whisper segments into video time and keep the ones with text."""
    rows: list[dict[str, Any]] = []
    for offset, resp in chunks:
        for seg in resp.get("segments") or []:
            text = str(seg.get("text") or "").strip()
            if text:
                rows.append({"start_s": round(float(seg.get("start", 0.0)) + offset, 2),
                             "end_s": round(float(seg.get("end", 0.0)) + offset, 2), "text": text})
    return rows


def transcribe(client: Any, video_path: Path, video_id: str, model: str) -> dict[str, Any]:
    pcm = extract_pcm16(video_path)
    duration_s = len(pcm) / (SAMPLE_RATE * 2)
    chunks: list[tuple[float, dict[str, Any]]] = []
    language = None
    for offset, part in chunk_pcm(pcm):
        if len(part) < SAMPLE_RATE * 2:  # under a second of audio
            continue
        data = wav_bytes(part)
        assert len(data) <= MAX_UPLOAD_BYTES, len(data)
        resp = client.audio.transcriptions.create(
            model=model, file=(f"{video_id}.wav", data), response_format="verbose_json",
            timestamp_granularities=["segment"])
        payload = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
        language = language or payload.get("language")
        chunks.append((offset, payload))
    return {"video_id": video_id, "language": language, "duration_s": round(duration_s, 2),
            "model": model, "segments": merge_segments(chunks)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video-dir", type=Path, required=True)
    ap.add_argument("--video-ids", type=Path, required=True, help="one video id per line")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--model", default="whisper-1")
    ap.add_argument("--keys-py", type=Path, default=Path("/fs/gamma-projects/vlm-robot/keys.py"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--base-url", default="https://us.api.openai.com/v1",
                    help="the project key is region-pinned; the default host answers 401 incorrect_hostname")
    args = ap.parse_args(argv)

    from openai import OpenAI
    client = OpenAI(api_key=load_openai_api_key(args.keys_py), base_url=args.base_url)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ids = [line.strip() for line in args.video_ids.read_text().splitlines() if line.strip()]

    def run(video_id: str) -> tuple[str, str]:
        cache = args.out_dir / f"{video_id}.json"
        if cache.exists():
            return video_id, "cached"
        video = args.video_dir / f"{video_id}.mp4"
        if not video.exists():
            return video_id, "missing video"
        try:
            result = transcribe(client, video, video_id, args.model)
        except Exception as exc:  # noqa: BLE001
            return video_id, f"error: {exc}"[:300]
        cache.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        return video_id, f"{len(result['segments'])} segments, {result['language']}, {result['duration_s']}s"

    done = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for video_id, status in pool.map(run, ids):
            done += 1
            print(f"[{done}/{len(ids)}] {video_id}: {status}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
