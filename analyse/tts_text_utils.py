import re
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

DEFAULT_LOG = print


def make_safe_slug(value: str) -> str:
    """Create a filesystem-friendly slug from the given text."""
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in value)
    safe = "_".join(filter(None, safe.split("_")))
    return safe or "input"


def load_input_texts(
    dir_candidates: Sequence[str],
    allowed_extensions: Optional[Iterable[str]],
    log: Callable[[str], None] = DEFAULT_LOG,
) -> Tuple[List[dict], Path]:
    """Return list of usable input files and the directory that held them.

    Each item includes path, slug, and raw text content.
    """
    log("[INFO] Resolving input text files...")
    input_dir: Optional[Path] = None
    for candidate in dir_candidates:
        candidate_path = Path(candidate)
        if candidate_path.is_dir():
            input_dir = candidate_path
            break

    if input_dir is None:
        raise FileNotFoundError("No input directory found. Place text files in @input or input.")

    allowed_set = {ext.lower() for ext in allowed_extensions} if allowed_extensions else None

    input_files = sorted(
        [path for path in input_dir.iterdir() if path.is_file()],
        key=lambda p: p.name.lower(),
    )

    inputs: List[dict] = []
    slug_counts: dict[str, int] = {}

    for input_path in input_files:
        if allowed_set and input_path.suffix.lower() not in allowed_set:
            log(f"[WARN] {input_path.name} skipped (extension not allowed: {input_path.suffix}).")
            continue

        try:
            text_content = input_path.read_text(encoding="utf-8").strip()
        except Exception as err:  # pragma: no cover - defensive logging
            log(f"[WARN] Failed to read {input_path}: {err} (skipped)")
            continue

        if not text_content:
            log(f"[WARN] {input_path.name} is empty after stripping; skipping.")
            continue

        base_slug = make_safe_slug(input_path.stem)
        counter = slug_counts.get(base_slug, 0)
        slug_counts[base_slug] = counter + 1
        slug = base_slug if counter == 0 else f"{base_slug}_{counter}"

        inputs.append(
            {
                "path": input_path,
                "slug": slug,
                "text": text_content,
            }
        )
        log(f"[INFO] Prepared input '{input_path.name}' as slug '{slug}' ({len(text_content)} chars).")

    if not inputs:
        expected_desc = "any file" if not allowed_set else f"extensions: {sorted(allowed_set)}"
        raise ValueError(f"No usable text files found in {input_dir} ({expected_desc}).")

    log(f"[INFO] Found {len(inputs)} usable input file(s) in {input_dir}.")
    return inputs, input_dir


_SENTENCE_PATTERN = re.compile(
    r"(.+?(?:[.!?]\"?|[.!?]['\")\]]+|[\n\r]+|$))",
    re.MULTILINE,
)


def _split_into_sentences(text: str) -> List[str]:
    sentences: List[str] = []
    for match in _SENTENCE_PATTERN.finditer(text):
        sentence = match.group(0).strip()
        if sentence:
            sentences.append(sentence)
    return sentences if sentences else [text.strip()]


def _split_long_sentence(sentence: str, max_chars: int) -> List[str]:
    words = sentence.split()
    if not words:
        return []

    parts: List[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if len(candidate) > max_chars and current:
            parts.append(current)
            current = word
        else:
            current = candidate
    parts.append(current)
    return parts


def chunk_text(text: str, max_chars: int, target_chars: Optional[int] = None) -> List[str]:
    """Split text into chunks <= max_chars, aiming for target_chars when possible."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if target_chars is not None and target_chars <= 0:
        raise ValueError("target_chars must be positive when provided")

    target = min(max_chars, target_chars) if target_chars else max_chars
    sentences = _split_into_sentences(text)
    chunks: List[str] = []
    current = ""

    def flush_current() -> None:
        nonlocal current
        if current:
            chunks.append(current)
            current = ""

    def add_part(part: str) -> None:
        nonlocal current
        part = part.strip()
        if not part:
            return

        if len(part) > max_chars:
            sub_parts = _split_long_sentence(part, max_chars)
            if len(sub_parts) == 1:
                for idx in range(0, len(part), max_chars):
                    add_part(part[idx : idx + max_chars])
                return
            for sub_part in sub_parts:
                add_part(sub_part)
            return

        while True:
            if not current:
                current = part
                if len(current) >= target:
                    flush_current()
                return

            candidate = f"{current} {part}"
            if len(candidate) <= max_chars:
                current = candidate
                if len(current) >= target:
                    flush_current()
                return

            flush_current()

    for sentence in sentences:
        parts = (
            _split_long_sentence(sentence, max_chars)
            if len(sentence) > max_chars
            else [sentence]
        )
        for part in parts:
            add_part(part)

    flush_current()

    return chunks
