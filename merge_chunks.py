import json
import re
from pathlib import Path

CHUNK_SIZE = 5
OVERLAP = 1
MIN_WORDS = 80
MAX_WORDS = 400

INPUT_DIR = Path("jsons")
OUTPUT_DIR = Path("merged_jsons")


def word_count(text: str) -> int:
    return len(text.split())


def clean_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def extract_metadata(data: dict, file_path: Path) -> dict:
    """
    Extract video metadata safely.

    Tries:
    1. top-level JSON fields
    2. nested metadata fields
    3. filename fallback
    """

    metadata = data.get("metadata", {}) if isinstance(data.get("metadata"), dict) else {}

    video_number = (
        data.get("video_number")
        or data.get("video_id")
        or data.get("number")
        or metadata.get("video_number")
        or metadata.get("video_id")
    )

    video_title = (
        data.get("video_title")
        or data.get("title")
        or data.get("video_name")
        or metadata.get("video_title")
        or metadata.get("title")
    )

    if video_number is None:
        match = re.match(r"(\d+)", file_path.stem)
        video_number = int(match.group(1)) if match else file_path.stem

    if not video_title:
        video_title = file_path.stem.replace("_", " ").replace("-", " ").strip()

    return {
        "video_number": video_number,
        "video_title": video_title,
        "source_file": file_path.name,
    }


def merge_chunk_group(chunk_group: list[dict], meta: dict) -> dict:
    return {
        "video_number": meta["video_number"],
        "video_title": meta["video_title"],
        "source_file": meta["source_file"],
        "start": chunk_group[0].get("start", 0),
        "end": chunk_group[-1].get("end", 0),
        "text": clean_text(" ".join(c.get("text", "") for c in chunk_group)),
        "source_count": len(chunk_group),
    }


def sliding_window_merge(chunks: list[dict], size: int, overlap: int) -> list[list[dict]]:
    if size <= 0:
        raise ValueError("CHUNK_SIZE must be greater than 0")

    if overlap >= size:
        raise ValueError("OVERLAP must be smaller than CHUNK_SIZE")

    step = size - overlap
    groups = []

    for i in range(0, len(chunks), step):
        group = chunks[i:i + size]
        if group:
            groups.append(group)

    return groups


def process_file(file_path: Path) -> tuple[int, list[str]]:
    warnings = []

    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    raw_chunks = data.get("chunks", [])

    raw_chunks = [
        chunk for chunk in raw_chunks
        if clean_text(chunk.get("text", ""))
    ]

    if not raw_chunks:
        return 0, ["No valid chunks found - skipped"]

    meta = extract_metadata(data, file_path)

    groups = sliding_window_merge(raw_chunks, CHUNK_SIZE, OVERLAP)
    merged_chunks = []

    pending_group = []

    for group in groups:
        pending_group.extend(group)

        merged = merge_chunk_group(pending_group, meta)
        words = word_count(merged["text"])

        if words < MIN_WORDS:
            continue

        if words > MAX_WORDS:
            warnings.append(
                f"Long chunk detected: words={words}, start={merged['start']:.1f}s"
            )

        merged_chunks.append(merged)
        pending_group = []

    if pending_group:
        merged = merge_chunk_group(pending_group, meta)

        if merged_chunks and word_count(merged["text"]) < MIN_WORDS:
            previous = merged_chunks[-1]
            previous["text"] = clean_text(previous["text"] + " " + merged["text"])
            previous["end"] = merged["end"]
            previous["source_count"] += merged["source_count"]
            warnings.append("Final short chunk absorbed into previous chunk")
        else:
            merged_chunks.append(merged)

    OUTPUT_DIR.mkdir(exist_ok=True)

    output_data = {
        "video_number": meta["video_number"],
        "video_title": meta["video_title"],
        "source_file": meta["source_file"],
        "total_chunks": len(merged_chunks),
        "chunks": merged_chunks,
        "full_text": data.get("text", ""),
    }

    out_path = OUTPUT_DIR / file_path.name

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    return len(merged_chunks), warnings


if __name__ == "__main__":
    json_files = sorted(INPUT_DIR.glob("*.json"))

    print(f"Found {len(json_files)} file(s)")
    print(f"CHUNK_SIZE={CHUNK_SIZE}, OVERLAP={OVERLAP}\n")

    total_chunks = 0

    for file_path in json_files:
        count, warnings = process_file(file_path)
        total_chunks += count

        status = "OK" if not warnings else "WARN"
        print(f"{status} {file_path.name} -> {count} merged chunks")

        for warning in warnings:
            print(f"  - {warning}")

    print(f"\nDone. Total merged chunks: {total_chunks}")
    print(f"Output saved to: {OUTPUT_DIR}/")