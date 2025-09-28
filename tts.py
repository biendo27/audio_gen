#!/usr/bin/env python3
import os
from pathlib import Path
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
REF_AUDIO = "resources/reference_edgar.mp3"
OUTPUT_DIR = "outputs_v2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

REF_SE_CACHE_DIR = os.path.join(OUTPUT_DIR, "cached_ref_se")
os.makedirs(REF_SE_CACHE_DIR, exist_ok=True)
REF_SE_CACHE_PATH = os.path.join(
    REF_SE_CACHE_DIR,
    os.path.splitext(os.path.basename(REF_AUDIO))[0] + "_se.pth"
)

SPEED = 1.0
WARMUP_TEXT = "Hello"

INPUT_DIR_CANDIDATES = ("@input", "input")
INPUT_EXTENSIONS = {".txt"}


def make_safe_slug(value: str) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in value)
    safe = "_".join(filter(None, safe.split("_")))
    return safe or "input"


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
# Extract reference speaker embedding (target SE) with caching
# -----------------------------
target_se = None
if os.path.exists(REF_SE_CACHE_PATH):
    try:
        print(f"[INFO] Loading cached reference SE from {REF_SE_CACHE_PATH} ...")
        target_se = torch.load(REF_SE_CACHE_PATH, map_location=device)
        print("[INFO] Reference SE loaded from cache.")
    except Exception as err:
        print(f"[WARN] Failed to load cached reference SE ({err}); re-extracting.")

if target_se is None:
    print(f"[INFO] Extracting reference SE from {REF_AUDIO} ...")
    target_se, _ = se_extractor.get_se(REF_AUDIO, tone_color_converter, vad=True)
    try:
        torch.save(target_se, REF_SE_CACHE_PATH)
        print(f"[INFO] Cached reference SE at {REF_SE_CACHE_PATH}.")
    except Exception as err:
        print(f"[WARN] Could not cache reference SE: {err}")
    print("[INFO] Reference SE extracted.")

# -----------------------------
# Load input text files
# -----------------------------
print("[INFO] Resolving input text files...")
INPUT_DIR = None
for candidate in INPUT_DIR_CANDIDATES:
    candidate_path = Path(candidate)
    if candidate_path.is_dir():
        INPUT_DIR = candidate_path
        break

if INPUT_DIR is None:
    raise SystemExit("[ERROR] No input directory found. Place text files in @input or input.")

input_files = sorted(
    [
        path
        for path in INPUT_DIR.iterdir()
        if path.is_file() and (not INPUT_EXTENSIONS or path.suffix.lower() in INPUT_EXTENSIONS)
    ],
    key=lambda p: p.name.lower()
)

inputs_to_process = []
slug_counts = {}

for input_path in input_files:
    try:
        text_content = input_path.read_text(encoding="utf-8").strip()
    except Exception as err:
        print(f"[WARN] Failed to read {input_path}: {err} (skipped)")
        continue

    if not text_content:
        print(f"[WARN] {input_path} is empty after stripping; skipping.")
        continue

    base_slug = make_safe_slug(input_path.stem)
    counter = slug_counts.get(base_slug, 0)
    slug_counts[base_slug] = counter + 1
    slug = base_slug if counter == 0 else f"{base_slug}_{counter}"

    inputs_to_process.append(
        {
            "path": input_path,
            "slug": slug,
            "text": text_content,
        }
    )
    print(f"[INFO] Prepared input '{input_path.name}' as slug '{slug}' ({len(text_content)} chars).")

expected_desc = "any file" if not INPUT_EXTENSIONS else f"extensions: {sorted(INPUT_EXTENSIONS)}"
if not inputs_to_process:
    raise SystemExit(f"[ERROR] No usable text files found in {INPUT_DIR} ({expected_desc}).")

print(f"[INFO] Found {len(inputs_to_process)} usable input file(s) in {INPUT_DIR}.")

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
skipped_speakers = set()

for input_meta in inputs_to_process:
    input_name = input_meta["path"].name
    slug = input_meta["slug"]
    text_payload = input_meta["text"]
    print(f"\n[INPUT] {input_name} -> slug={slug} ({len(text_payload)} chars)")

    for spk_key, spk_id in speaker_ids.items():
        if spk_key not in source_se_map:
            if spk_key not in skipped_speakers:
                print(f"[SKIP] {spk_key} skipped because SE not available.")
                skipped_speakers.add(spk_key)
            continue

        print(f"[RUN] {spk_key} -> id={spk_id}")
        tmp_filename = f"tmp_{slug}_{spk_key}.wav"
        output_filename = f"output_v2_{slug}_{spk_key}.wav"
        src_path = os.path.join(OUTPUT_DIR, tmp_filename)
        save_path = os.path.join(OUTPUT_DIR, output_filename)

        # 1) generate TTS (should be fast because warm-up done)
        t0 = time.time()
        try:
            model.tts_to_file(text_payload, spk_id, src_path, speed=SPEED)
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
        print(f"   [INFO] Generation time: {gen_time:.2f}s")
        print(f"   [INFO] Audio duration: {audio_duration if audio_duration else 'unknown'}s")
        print(f"   [INFO] RTF: {rtf:.2f}")

        # 3) convert tone color to reference
        try:
            tone_color_converter.convert(
                audio_src_path=src_path,
                src_se=source_se_map[spk_key],
                tgt_se=target_se,
                output_path=save_path,
                message="@MyShell"
            )
            print(f"   [INFO] Saved converted file: {save_path}")
        except Exception as e:
            print(f"   [ERROR] tone_color_converter.convert failed for {spk_key}: {e}")

print("\n[INFO] All done.")

