#!/usr/bin/env python3
"""
Interactive QA -> Gemini -> EN-US TTS (OpenVoice Melo) -> ToneColorConverter clone voice
Works even if google-generativeai is not installed / API key missing (fallback demo reply).
"""
import os
import time
import sys
import torch
import torchaudio
from dotenv import load_dotenv

# OpenVoice imports (correct)
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from melo.api import TTS

# ---- config ----
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()  # put key in .env if you want real Gemini
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
CKPT_CONVERTER_DIR = "checkpoints_v2/converter"
CONVERTER_CONFIG = os.path.join(CKPT_CONVERTER_DIR, "config.json")
CONVERTER_CKPT = os.path.join(CKPT_CONVERTER_DIR, "checkpoint.pth")
BASE_SPK_DIR = "checkpoints_v2/base_speakers/ses"
REF_AUDIO = "resources/example_reference.mp3"
OUT_DIR = "outputs_v2"
os.makedirs(OUT_DIR, exist_ok=True)

SPEED = 1.0

# ---- try import google generative ai (Gemini) ----
_HAS_GENAI = False
try:
    import google.generativeai as genai
    _HAS_GENAI = True
except Exception:
    _HAS_GENAI = False

# helper: find EN-US speaker id robustly from mapping-like object
def find_en_us_speaker(mapping):
    # mapping usually model.hps.data.spk2id
    try:
        if hasattr(mapping, "items"):
            for k, v in mapping.items():
                k_clean = str(k).lower().replace("_", "-")
                if "en" in k_clean and "us" in k_clean:
                    return v, k
    except Exception:
        pass
    # fallback exact keys
    for cand in ("EN-US", "EN_US", "en-us", "en_us", "EN-US".lower()):
        if cand in mapping:
            return mapping[cand], cand
    return None, None

# ---- check converter files ----
if not os.path.exists(CONVERTER_CONFIG) or not os.path.exists(CONVERTER_CKPT):
    print(f"[ERROR] Converter config/ckpt missing. Expecting:\n  {CONVERTER_CONFIG}\n  {CONVERTER_CKPT}")
    sys.exit(1)

print(f"[INFO] Device: {DEVICE}")
print("[INFO] Loading ToneColorConverter (config -> ckpt)...")
tone_color_converter = ToneColorConverter(CONVERTER_CONFIG, device=DEVICE)
tone_color_converter.load_ckpt(CONVERTER_CKPT)
print("[INFO] ToneColorConverter loaded.")

# ---- extract target SE from reference audio (the voice we clone) ----
if not os.path.exists(REF_AUDIO):
    print(f"[ERROR] reference audio not found: {REF_AUDIO}")
    sys.exit(1)

print(f"[INFO] Extracting reference SE from {REF_AUDIO} (this may take a moment)...")
target_se, _ = se_extractor.get_se(REF_AUDIO, tone_color_converter, vad=True)
print("[INFO] Reference SE extracted.")

# ---- preload TTS EN model once ----
print("[INFO] Loading EN TTS model (Melo.TTS)...")
tts_model = TTS(language="EN", device=DEVICE)  # loads tokenizer/BERT once if needed
spk_map = tts_model.hps.data.spk2id
print("[INFO] Available speakers:", list(spk_map.keys()))

speaker_id, speaker_key = find_en_us_speaker(spk_map)
if speaker_id is None:
    print("[ERROR] Could not find EN-US speaker in model.hps.data.spk2id. Available keys:")
    print(list(spk_map.keys()))
    sys.exit(1)
print(f"[INFO] Using speaker: {speaker_key} -> id={speaker_id}")

# ---- load source speaker embedding for EN-US (base speaker) ----
se_filename = speaker_key.lower().replace("_", "-").replace(" ", "-")
src_se_path = os.path.join(BASE_SPK_DIR, f"{se_filename}.pth")
if not os.path.exists(src_se_path):
    print(f"[ERROR] source speaker SE file not found: {src_se_path}")
    sys.exit(1)
source_se = torch.load(src_se_path, map_location=DEVICE)
print(f"[INFO] Loaded source SE from {src_se_path}")

# ---- warm-up once to eliminate cold-start delay ----
print("[INFO] Warm-up TTS (single short utterance)...")
warmup_tmp = os.path.join(OUT_DIR, "warmup.wav")
try:
    tts_model.tts_to_file("Hello", speaker_id, warmup_tmp, speed=1.0)
    # remove warmup file
    try:
        os.remove(warmup_tmp)
    except Exception:
        pass
except Exception as e:
    print("[WARN] Warm-up failed:", e)
print("[INFO] Warm-up done — subsequent TTS should be fast.")

# ---- configure Gemini if available + key present ----
_USE_GEMINI = False
if _HAS_GENAI and GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        _USE_GEMINI = True
        print("[INFO] google.generativeai available and configured (Gemini will be used).")
    except Exception as e:
        print("[WARN] Could not configure google.generativeai:", e)
        _USE_GEMINI = False
else:
    if not _HAS_GENAI:
        print("[WARN] google.generativeai not installed — Gemini calls will fallback to demo reply.")
    elif not GOOGLE_API_KEY:
        print("[WARN] GOOGLE_API_KEY missing in .env — Gemini calls will fallback to demo reply.")

def ask_gemini(question: str) -> str:
    """Return text answer from Gemini if configured, otherwise demo reply."""
    if not _USE_GEMINI:
        return f"(demo reply) I received your question: {question}"
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(question)
        # response.text is typical; but ensure safe fallback
        if hasattr(response, "text"):
            return response.text.strip()
        elif isinstance(response, dict) and "candidates" in response:
            return response["candidates"][0].get("content", "").strip()
        else:
            return str(response)
    except Exception as e:
        print("[WARN] Gemini call failed:", e)
        return f"(gemini error) {e}"

# ---- interactive loop ----
print("\n=== Interactive: Ask -> Gemini -> EN-US TTS (cloned) ===")
print("Type 'exit' to quit.\n")
count = 0
while True:
    try:
        q = input("Bạn muốn hỏi gì? > ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n[INFO] Exiting.")
        break
    if not q:
        continue
    if q.lower() in ("exit", "quit", "bye"):
        print("[INFO] Bye!")
        break

    print("[INFO] Asking Gemini...")
    answer = ask_gemini(q)
    print("[GEMINI]", answer)

    # generate TTS -> tmp source wav
    tmp_src = os.path.join(OUT_DIR, f"tmp_{count}.wav")
    out_final = os.path.join(OUT_DIR, f"answer_{count}.wav")

    try:
        t0 = time.time()
        tts_model.tts_to_file(answer, speaker_id, tmp_src, speed=SPEED)
        gen_time = time.time() - t0
    except Exception as e:
        print("[ERROR] tts_to_file failed:", e)
        continue

    # convert tone/voice to reference
    try:
        tone_color_converter.convert(
            audio_src_path=tmp_src,
            src_se=source_se,
            tgt_se=target_se,
            output_path=out_final,
            message="@MyShell"
        )
    except Exception as e:
        print("[ERROR] tone_color_converter.convert failed:", e)
        # keep tmp_src for debugging
        continue

    # measure duration (robust)
    duration = None
    try:
        info = torchaudio.info(out_final)
        duration = info.num_frames / info.sample_rate
    except Exception:
        try:
            w, sr = torchaudio.load(out_final)
            duration = w.size(1) / sr
        except Exception:
            duration = None

    print(f"✅ Saved: {out_final}  (gen_time={gen_time:.2f}s, duration={duration:.2f}s if available)")
    count += 1
