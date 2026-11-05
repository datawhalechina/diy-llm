"""Fetch the labeled M3KE dev examples used by the lm-eval smoke task."""

from __future__ import annotations

import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

from datasets import load_dataset

ARCHIVE_URL = (
    "https://raw.githubusercontent.com/tjunlp-lab/M3KE/"
    "b370aae92167610791c34d16ef1b5a3e41805e62/data/M3KE.zip"
)
ARCHIVE_SHA256 = "02e6f4cb67338d10013d3d334c871aac44e65c5def469428a7707cbc5b88d29d"
SOURCE_MEMBER = "dev/Chinese-Arts & Humanities-High school.jsonl"
OUTPUT_FILE = Path(__file__).parent / "data" / "m3ke_chinese_high_school_dev.jsonl"
LABELS = ("A", "B", "C", "D")


def _validate_rows(rows: list[dict]) -> None:
    if len(rows) != 5:
        raise ValueError(f"Expected the 5 labeled M3KE dev examples, got {len(rows)}.")
    for index, row in enumerate(rows):
        missing = [key for key in ("question", *LABELS, "answer") if key not in row]
        if missing:
            raise ValueError(f"Example {index} is missing fields: {missing}.")
        if row["answer"].strip() not in LABELS:
            raise ValueError(
                f"Example {index} has an unlabeled or invalid answer: {row['answer']!r}."
            )


def _load_rows() -> list[dict]:
    request = urllib.request.Request(
        ARCHIVE_URL, headers={"User-Agent": "diy-llm-m3ke-eval/1.0"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        archive_bytes = response.read()

    actual_hash = hashlib.sha256(archive_bytes).hexdigest()
    if actual_hash != ARCHIVE_SHA256:
        raise ValueError(
            f"M3KE archive checksum mismatch: expected {ARCHIVE_SHA256}, got {actual_hash}."
        )

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        payload = archive.read(SOURCE_MEMBER)
    rows = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line]
    _validate_rows(rows)
    return rows


def ensure_dev_file() -> Path:
    if not OUTPUT_FILE.exists():
        rows = _load_rows()
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_FILE.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
    else:
        with OUTPUT_FILE.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        _validate_rows(rows)
    return OUTPUT_FILE


def load_m3ke_dev(**_kwargs):
    """Return the labeled development split in lm-eval's expected split mapping."""
    path = ensure_dev_file()
    return load_dataset("json", data_files={"validation": str(path)})


if __name__ == "__main__":
    path = ensure_dev_file()
    print(f"Prepared {path} with 5 labeled M3KE dev examples.")
