import os
import re
import string
import librosa
import numpy as np
from jiwer import wer, cer
from phonemizer import phonemize
from faster_whisper import WhisperModel
import torch

import faster_whisper, ctranslate2, huggingface_hub
print("faster_whisper:", faster_whisper.__version__)
import ctranslate2; print("ctranslate2:", ctranslate2.__version__)
import huggingface_hub; print("huggingface_hub:", huggingface_hub.__version__)


# ----------------------------
# Utility: normalize text
# ----------------------------
_punct_table = str.maketrans({k: " " for k in string.punctuation})
def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = s.translate(_punct_table)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

# ----------------------------
# Phoneme error rate
# ----------------------------
def levenshtein(a, b):
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            tmp = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(
                dp[j] + 1,      # deletion
                dp[j - 1] + 1,  # insertion
                prev + cost,    # substitution
            )
            prev = tmp
    return dp[n]

def phoneme_error_rate(ref_ph, hyp_ph):
    if not ref_ph:
        return 1.0 if hyp_ph else 0.0
    dist = levenshtein(ref_ph, hyp_ph)
    return dist / max(1, len(ref_ph))

# ----------------------------
# Prosody analysis
# ----------------------------
def analyze_prosody(audio_path, transcript, lang="en"):
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    # pitch
    f0 = librosa.yin(y, fmin=50, fmax=600, sr=sr)
    f0_valid = f0[np.isfinite(f0)]
    pitch_median = float(np.median(f0_valid)) if len(f0_valid) > 0 else 0
    pitch_stdev = float(np.std(f0_valid)) if len(f0_valid) > 0 else 0
    # energy
    rms = librosa.feature.rms(y=y)[0]
    energy_mean = float(np.mean(rms))
    duration_s = len(y) / sr
    # syllable estimate (count vowels)
    vowels = "aeiou" if lang == "en" else "aeiouy"
    syllables = sum([sum(ch in vowels for ch in word) > 0 for word in transcript.split()])
    speech_rate = syllables / max(0.1, duration_s)
    return {
        "pitch_median": pitch_median,
        "pitch_stdev": pitch_stdev,
        "energy_mean": energy_mean,
        "duration_s": duration_s,
        "syllables_est": syllables,
        "speech_rate": speech_rate,
    }

# ----------------------------
# Evaluation with faster_whisper
# ----------------------------
def evaluate(audio_path, reference_text=None, lang="en"):
    # chọn device và precision phù hợp
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    faster_whisper_model = "nyrahealth/faster_CrisperWhisper"
    model = WhisperModel(faster_whisper_model,
                         device=device,
                         compute_type=compute_type)

    # run ASR
    segments, info = model.transcribe(audio_path,
                                      beam_size=1,
                                      language=lang,
                                      word_timestamps=True,
                                      without_timestamps=False)

    hyp_text = normalize_text(" ".join([seg.text for seg in segments]))
    ref_text = normalize_text(reference_text) if reference_text else None

    # metrics
    w = c = p = None
    if ref_text:
        w = wer(ref_text, hyp_text)
        c = cer(ref_text, hyp_text)
        phoneme_lang = "en-us" if lang.startswith("en") else lang
        ref_ph = phonemize(ref_text, backend="espeak", language=phoneme_lang).split()
        hyp_ph = phonemize(hyp_text, backend="espeak", language=phoneme_lang).split()
        p = phoneme_error_rate(ref_ph, hyp_ph)

    # prosody
    prosody = analyze_prosody(audio_path, hyp_text, lang)

    # report
    print("\n--- Pronunciation Report ---")
    print(f"ASR: {hyp_text}")
    if ref_text:
        print(f"Ref: {ref_text}")
        print(f"WER: {w:.3f}, CER: {c:.3f}, PER: {p:.3f}")
    print("Prosody:", prosody)

# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    audio = "content/stutter_audio.wav"
    ref = "The quick brown fox jumps over the lazy dog you know."
    evaluate(audio, ref, "en")
