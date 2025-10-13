#!/usr/bin/env python3
import os
from pathlib import Path
import sys
import time
import torch
import torchaudio

from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from melo.api import TTS
from analyse.tts_config import (
    MAX_CHARS_PER_CHUNK,
    TARGET_CHARS_PER_CHUNK,
    VOICE_STYLE_WHITELIST,
    PRESERVE_CHUNK_OUTPUTS,
)
from analyse.tts_text_utils import load_input_texts, chunk_text
from analyse.tts_audio_utils import concatenate_wavs, safe_remove_files


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
# Disable watermark to save GPU memory unless explicitly needed.
tone_color_converter.watermark_model = None
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
# Load and chunk input text files
# -----------------------------
try:
    inputs_to_process, input_dir = load_input_texts(
        INPUT_DIR_CANDIDATES,
        INPUT_EXTENSIONS,
        log=print,
    )
except (FileNotFoundError, ValueError) as err:
    raise SystemExit(f"[ERROR] {err}") from err

for meta in inputs_to_process:
    chunks = chunk_text(meta["text"], MAX_CHARS_PER_CHUNK, TARGET_CHARS_PER_CHUNK)
    meta["chunks"] = chunks
    print(
        f"[INFO] {meta['path'].name} chunked into {len(chunks)} segment(s) "
        f"(limit={MAX_CHARS_PER_CHUNK} chars)."
    )

# -----------------------------
# Preload EN TTS model once (this will load tokenizer/BERT if needed)
# -----------------------------
print("[INFO] Preloading EN TTS model...")
model = TTS(language="EN", device=device)
speaker_ids = model.hps.data.spk2id  # mapping-like
print("[INFO] EN model ready. Available speakers:", list(speaker_ids.keys()))

# -----------------------------
# Resolve desired voice styles
# -----------------------------
if VOICE_STYLE_WHITELIST:
    selected_speakers = {}
    for spk_key in VOICE_STYLE_WHITELIST:
        if spk_key in speaker_ids:
            selected_speakers[spk_key] = speaker_ids[spk_key]
        else:
            print(f"[WARN] Requested voice '{spk_key}' not found in model (skipped).")
    if not selected_speakers:
        raise SystemExit("[ERROR] No valid voices resolved from VOICE_STYLE_WHITELIST.")
else:
    selected_speakers = dict(speaker_ids)

print(f"[INFO] Using {len(selected_speakers)} voice style(s) for synthesis.")

# -----------------------------
# Preload source SE files for selected speakers (fail early if missing)
# -----------------------------
print("[INFO] Preloading base speaker SE files...")
source_se_map = {}
for spk_key, spk_id in selected_speakers.items():
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
for spk_key, spk_id in selected_speakers.items():
    if spk_key not in source_se_map:
        continue
    warmup_path = os.path.join(OUTPUT_DIR, f"warmup_{spk_key}.wav")
    try:
        model.tts_to_file(WARMUP_TEXT, spk_id, warmup_path, speed=1.0)
        try:
            os.remove(warmup_path)
        except Exception:
            pass
    except Exception as e:
        print(f"  - [WARN] Warm-up failed for {spk_key}: {e}")
print("[INFO] Warm-up done.")

# -----------------------------
# Generation loop with chunking and merging
# -----------------------------
print("[INFO] Start generation loop...")
skipped_speakers = set()

for meta in inputs_to_process:
    input_name = meta["path"].name
    slug = meta["slug"]
    chunks = meta.get("chunks", [])
    print(f"\n[INPUT] {input_name} -> slug={slug} ({len(chunks)} chunk(s))")

    for spk_key, spk_id in selected_speakers.items():
        if spk_key not in source_se_map:
            if spk_key not in skipped_speakers:
                print(f"[SKIP] {spk_key} skipped because SE not available.")
                skipped_speakers.add(spk_key)
            continue

        print(f"[VOICE] {spk_key} -> id={spk_id}")
        chunk_tmp_paths = []
        chunk_output_paths = []
        speaker_failed = False

        for idx, chunk_text_payload in enumerate(chunks, start=1):
            chunk_tag = f"{slug}_chunk{idx:03d}"
            tmp_filename = f"tmp_{chunk_tag}_{spk_key}.wav"
            output_filename = f"output_v2_{chunk_tag}_{spk_key}.wav"
            src_path = os.path.join(OUTPUT_DIR, tmp_filename)
            save_path = os.path.join(OUTPUT_DIR, output_filename)

            print(f"  [CHUNK] {chunk_tag} ({len(chunk_text_payload)} chars)")
            t0 = time.time()
            try:
                model.tts_to_file(chunk_text_payload, spk_id, src_path, speed=SPEED)
            except Exception as e:
                speaker_failed = True
                print(f"   [ERROR] tts_to_file failed for {spk_key} on {chunk_tag}: {e}")
                safe_remove_files([src_path], log=print)
                break
            gen_time = time.time() - t0

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

            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
            try:
                tone_color_converter.convert(
                    audio_src_path=src_path,
                    src_se=source_se_map[spk_key],
                    tgt_se=target_se,
                    output_path=save_path,
                    message="@MyShell"
                )
                print(f"   [INFO] Saved converted chunk: {save_path}")
            except Exception as e:
                speaker_failed = True
                print(f"   [ERROR] tone_color_converter.convert failed for {spk_key} on {chunk_tag}: {e}")
                safe_remove_files([src_path, save_path], log=print)
                break
            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()

            chunk_tmp_paths.append(src_path)
            chunk_output_paths.append(save_path)

        if speaker_failed:
            safe_remove_files(chunk_tmp_paths + chunk_output_paths, log=print)
            continue

        if not chunk_output_paths:
            print(f"  [WARN] No chunks rendered for {spk_key}; skipping merge.")
            continue

        final_filename = f"output_v2_{slug}_{spk_key}.wav"
        final_path = os.path.join(OUTPUT_DIR, final_filename)
        try:
            concatenate_wavs(chunk_output_paths, final_path)
            print(f"  [INFO] Merged {len(chunk_output_paths)} chunk(s) into {final_path}")
        except Exception as err:
            print(f"  [ERROR] Failed to merge chunks for {spk_key}: {err}")
            safe_remove_files([final_path], log=print)

        if not PRESERVE_CHUNK_OUTPUTS:
            safe_remove_files(chunk_tmp_paths + chunk_output_paths, log=print)

print("\n[INFO] All done.")
