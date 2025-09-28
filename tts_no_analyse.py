#!/usr/bin/env python3
import os
import time
import torch
import torchaudio

from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from melo.api import TTS

# -----------------------------
# Config paths
# -----------------------------
CKPT_CONVERTER_DIR = "checkpoints_v2/converter"
CONVERTER_CONFIG = os.path.join(CKPT_CONVERTER_DIR, "config.json")
CONVERTER_CKPT = os.path.join(CKPT_CONVERTER_DIR, "checkpoint.pth")

BASE_SPEAKERS_DIR = "checkpoints_v2/base_speakers/ses"
REF_AUDIO = "resources/reference_wiliam.mp3"
OUTPUT_DIR = "outputs_v2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TEXT = ("Technology is changing our world faster than ever before. Every day, new ideas and innovations shape the way we live, work, and connect with each other."
        "From smartphones to artificial intelligence, new inventions make our lives easier and more connected.")
SPEED = 1.0
WARMUP_TEXT = "Hello"

# -----------------------------
# Device selection
# -----------------------------
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
print(f"[INFO] Using device: {device}")

# -----------------------------
# Validate converter files
# -----------------------------
if not os.path.exists(CONVERTER_CONFIG):
    raise SystemExit(f"[ERROR] converter config not found: {CONVERTER_CONFIG}")
if not os.path.exists(CONVERTER_CKPT):
    raise SystemExit(f"[ERROR] converter checkpoint not found: {CONVERTER_CKPT}")

# -----------------------------
# Load ToneColorConverter (correct usage)
# -----------------------------
print("[INFO] Loading ToneColorConverter (config then ckpt)...")
tone_color_converter = ToneColorConverter(CONVERTER_CONFIG, device=device)
tone_color_converter.load_ckpt(CONVERTER_CKPT)
print("[INFO] ToneColorConverter loaded.")

# -----------------------------
# Extract reference speaker embedding (target SE)
# -----------------------------
print(f"[INFO] Extracting reference SE from {REF_AUDIO} ...")
target_se, _ = se_extractor.get_se(REF_AUDIO, tone_color_converter, vad=True)
print("[INFO] Reference SE extracted.")

# -----------------------------
# Preload EN TTS model once (this will load tokenizer/BERT if needed)
# -----------------------------
print("[INFO] Preloading EN TTS model...")
model = TTS(language="EN", device=device)
speaker_ids = model.hps.data.spk2id  # mapping-like
print("[INFO] EN model ready. Available speakers:", list(speaker_ids.keys()))

# -----------------------------
# Preload source SE files for speakers (fail early if missing)
# -----------------------------
print("[INFO] Preloading base speaker SE files...")
source_se_map = {}
for spk_key in list(speaker_ids.keys()):
    filename = str(spk_key).lower().replace("_", "-").replace(" ", "-")
    se_path = os.path.join(BASE_SPEAKERS_DIR, f"{filename}.pth")
    try:
        src_se = torch.load(se_path, map_location=device)
        source_se_map[spk_key] = src_se
        print(f"  - Loaded SE for {spk_key} from {se_path}")
    except FileNotFoundError:
        print(f"  - [WARN] SE file not found for {spk_key}: expected {se_path} (will skip)")
    except Exception as e:
        print(f"  - [WARN] Failed to load SE for {spk_key}: {e} (will skip)")

# -----------------------------
# Warm-up (run once to avoid cold-start)
# -----------------------------
print("[INFO] Warming up model (this avoids cold-start delay)...")
for spk_key, spk_id in speaker_ids.items():
    if spk_key not in source_se_map:
        # skip speakers without SE
        continue
    warmup_path = os.path.join(OUTPUT_DIR, f"warmup_{spk_key}.wav")
    try:
        model.tts_to_file(WARMUP_TEXT, spk_id, warmup_path, speed=1.0)
        # optionally delete warmup file to avoid clutter:
        try:
            os.remove(warmup_path)
        except Exception:
            pass
    except Exception as e:
        print(f"  - [WARN] Warm-up failed for {spk_key}: {e}")
print("[INFO] Warm-up done.")

# -----------------------------
# Generation loop (fast now)
# -----------------------------
print("[INFO] Start generation loop...")
for spk_key, spk_id in speaker_ids.items():
    if spk_key not in source_se_map:
        print(f"[SKIP] {spk_key} skipped because SE not available.")
        continue

    print(f"\n[RUN] {spk_key} -> id={spk_id}")
    src_path = os.path.join(OUTPUT_DIR, f"tmp_{spk_key}.wav")
    save_path = os.path.join(OUTPUT_DIR, f"output_v2_{spk_key}.wav")

    # 1) generate TTS (should be fast because warm-up done)
    t0 = time.time()
    try:
        model.tts_to_file(TEXT, spk_id, src_path, speed=SPEED)
    except Exception as e:
        print(f"   [ERROR] tts_to_file failed for {spk_key}: {e}")
        continue
    gen_time = time.time() - t0

    # 2) measure duration robustly
    audio_duration = None
    try:
        info = torchaudio.info(src_path)
        audio_duration = info.num_frames / info.sample_rate
    except Exception:
        try:
            waveform, sr = torchaudio.load(src_path)
            audio_duration = waveform.size(1) / sr
        except Exception as e:
            print(f"   [WARN] Could not measure duration for {src_path}: {e}")

    rtf = (gen_time / audio_duration) if (audio_duration and audio_duration > 0) else float("inf")
    print(f"   ⏱ Generation time: {gen_time:.2f}s")
    print(f"   🎵 Audio duration: {audio_duration if audio_duration else 'unknown'}s")
    print(f"   ⚡ RTF: {rtf:.2f}")

    # 3) convert tone color to reference
    try:
        tone_color_converter.convert(
            audio_src_path=src_path,
            src_se=source_se_map[spk_key],
            tgt_se=target_se,
            output_path=save_path,
            message="@MyShell"
        )
        print(f"   ✅ Saved converted file: {save_path}")
    except Exception as e:
        print(f"   [ERROR] tone_color_converter.convert failed for {spk_key}: {e}")

print("\n[INFO] All done.")

