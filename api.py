#!/usr/bin/env python3
import os
import sys
import time
import uuid
import shutil
from typing import Any, Dict, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

import torch
import librosa
from dotenv import load_dotenv
import wave
import struct

# STT (based on testSTT.py approach)
from faster_whisper import WhisperModel

# TTS (based on test2.py approach)
from melo.api import TTS


# -----------------------------
# Config
# -----------------------------
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
UPLOAD_DIR = os.path.join(PROJECT_ROOT, "uploads")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs_api")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

STATIC_PREFIX = "/static"

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
GEN_MAX_TOKENS = int(os.getenv("GEN_MAX_TOKENS", "128"))
GEN_TEMPERATURE = float(os.getenv("GEN_TEMPERATURE", "0.7"))

# Device selection (like test2.py)
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda:0"
else:
    DEVICE = "cpu"

# STT model (inspired by testSTT.py defaults)
ASR_MODEL = os.getenv("ASR_MODEL", "nyrahealth/faster_CrisperWhisper").strip()
ASR_BEAM_SIZE = int(os.getenv("ASR_BEAM_SIZE", "1"))

# Optional Gemini
_HAS_GENAI = False
try:
    import google.generativeai as genai
    _HAS_GENAI = True
except Exception:
    _HAS_GENAI = False


# -----------------------------
# Helpers
# -----------------------------
def find_en_us_speaker(mapping: Any):
    try:
        if hasattr(mapping, "items"):
            for k, v in mapping.items():
                k_clean = str(k).lower().replace("_", "-")
                if "en" in k_clean and "us" in k_clean:
                    return v, k
    except Exception:
        pass
    for cand in ("EN-US", "EN_US", "en-us", "en_us", "EN-US".lower()):
        if cand in mapping:
            return mapping[cand], cand
    try:
        if hasattr(mapping, "items"):
            k0, v0 = next(iter(mapping.items()))
            return v0, k0
    except Exception:
        pass
    return None, None


def measure_duration_seconds(audio_path: str) -> Optional[float]:
    try:
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        return float(len(y)) / float(sr) if sr and len(y) else None
    except Exception:
        return None


def ask_gemini(prompt_text: str) -> str:
    if not (_HAS_GENAI and GOOGLE_API_KEY):
        return f"(demo reply) I received your question: {prompt_text}"
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel(
            "gemini-1.5-flash",
            generation_config={
                "max_output_tokens": GEN_MAX_TOKENS,
                "temperature": GEN_TEMPERATURE,
            },
        )
        response = model.generate_content(prompt_text, generation_config={
            "max_output_tokens": GEN_MAX_TOKENS,
            "temperature": GEN_TEMPERATURE,
        })
        if hasattr(response, "text"):
            return response.text.strip()
        return str(response)
    except Exception as e:
        return f"(gemini error) {e}"


def _generate_silence_wav(file_path: str, duration_ms: int = 500, sample_rate: int = 16000) -> None:
    """Generate a short silent wav file for warm-up without external deps."""
    num_samples = int(sample_rate * (duration_ms / 1000.0))
    with wave.open(file_path, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(sample_rate)
        silence_frame = struct.pack('<h', 0)
        wf.writeframes(silence_frame * num_samples)


# -----------------------------
# Initialize heavy models once
# -----------------------------
print(f"[INFO] Device: {DEVICE}")

print(f"[INFO] Loading faster-whisper: {ASR_MODEL} ...")
compute_type = "float16" if DEVICE.startswith("cuda") else ("int8" if DEVICE == "cpu" else "float32")
whisper_model = WhisperModel(ASR_MODEL, device=("cuda" if DEVICE.startswith("cuda") else DEVICE), compute_type=compute_type)
print("[INFO] WhisperModel ready.")

print("[INFO] Loading Melo TTS (EN) ...")
tts_model = TTS(language="EN", device=DEVICE)
spk_map = tts_model.hps.data.spk2id
speaker_id, speaker_key = find_en_us_speaker(spk_map)
if speaker_id is None:
    raise SystemExit("[ERROR] Could not find a suitable speaker id in MeloTTS model.")
print(f"[INFO] TTS ready. Default speaker: {speaker_key} -> id={speaker_id}")

# --- Warm-up STT & TTS to avoid first-call latency ---
try:
    # STT warm-up with a tiny generated silent wav
    warmup_in = os.path.join(OUTPUT_DIR, "_warmup_in.wav")
    _generate_silence_wav(warmup_in, duration_ms=200)
    list(whisper_model.transcribe(warmup_in, beam_size=1, word_timestamps=False)[0])
except Exception:
    pass
try:
    # TTS warm-up with a short phrase
    warmup_out = os.path.join(OUTPUT_DIR, "_warmup_out.wav")
    tts_model.tts_to_file("Hello", speaker_id, warmup_out, speed=1.0)
    try:
        os.remove(warmup_out)
    except Exception:
        pass
except Exception:
    pass


# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="OpenVoice API: STT -> Gemini -> TTS",
              version="0.1.0",
              description="Upload audio, transcribe (faster-whisper), ask Gemini, synthesize reply (MeloTTS). Returns timings.")

app.mount(STATIC_PREFIX, StaticFiles(directory=OUTPUT_DIR), name="static")


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "device": DEVICE,
        "asr_model": ASR_MODEL,
        "gemini": bool(_HAS_GENAI and GOOGLE_API_KEY),
        "tts_language": "EN",
        "default_speaker": str(speaker_key),
    }


@app.get("/speakers")
def speakers() -> Dict[str, Any]:
    try:
        keys = list(spk_map.keys())
    except Exception:
        keys = []
    return {"speakers": keys}


@app.get("/download/{filename}")
def download_file(filename: str):
    # prevent path traversal
    safe_name = os.path.basename(filename)
    file_path = os.path.join(OUTPUT_DIR, safe_name)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(file_path, media_type="application/octet-stream", filename=safe_name)


@app.post("/stt-gemini-tts")
async def stt_gemini_tts(
    file: UploadFile = File(...),
    speed: float = Form(1.0),
) -> JSONResponse:
    if speed <= 0:
        raise HTTPException(status_code=400, detail="speed must be > 0")

    req_id = uuid.uuid4().hex[:8]
    in_name = f"{int(time.time())}_{req_id}_{file.filename}"
    in_path = os.path.join(UPLOAD_DIR, in_name)

    try:
        with open(in_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to save upload: {e}")
    finally:
        try:
            await file.close()
        except Exception:
            pass

    timings_ms: Dict[str, float] = {}
    t_total0 = time.time()

    # --- STT ---
    t0 = time.time()
    try:
        segments, info = whisper_model.transcribe(
            in_path,
            beam_size=ASR_BEAM_SIZE,
            word_timestamps=True,
            without_timestamps=False,
        )
        stt_text = " ".join([seg.text for seg in segments]).strip()
        detected_language = getattr(info, "language", None)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT failed: {e}")
    timings_ms["stt_ms"] = (time.time() - t0) * 1000.0

    # --- LLM (Gemini) ---
    t0 = time.time()
    answer_text = ask_gemini(stt_text)
    timings_ms["llm_ms"] = (time.time() - t0) * 1000.0

    # --- TTS ---
    out_name = f"reply_{int(time.time())}_{req_id}.wav"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    t0 = time.time()
    try:
        tts_model.tts_to_file(answer_text, speaker_id, out_path, speed=speed)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS failed: {e}")
    timings_ms["tts_ms"] = (time.time() - t0) * 1000.0

    total_ms = (time.time() - t_total0) * 1000.0
    timings_ms["total_ms"] = total_ms

    audio_duration_s = measure_duration_seconds(out_path)

    return JSONResponse({
        "status": "ok",
        "input_file": os.path.basename(in_path),
        "stt_text": stt_text,
        "detected_language": detected_language,
        "llm_answer": answer_text,
        "audio_file": os.path.basename(out_path),
        "audio_url": f"{STATIC_PREFIX}/{out_name}",
        "download_url": f"/download/{out_name}",
        "audio_duration_s": audio_duration_s,
        "timings_ms": {k: round(v, 2) for k, v in timings_ms.items()},
        "device": DEVICE,
        "models": {
            "asr": ASR_MODEL,
            "tts_language": "EN",
            "speaker": str(speaker_key),
        },
    })


if __name__ == "__main__":
    try:
        import uvicorn
    except Exception:
        print("[ERROR] uvicorn not installed. Install with: pip install uvicorn[standard] fastapi", file=sys.stderr)
        sys.exit(1)
    uvicorn.run("api:app", host="0.0.0.0", port=8686, reload=False)


