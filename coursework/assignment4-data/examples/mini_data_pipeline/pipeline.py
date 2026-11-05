#!/usr/bin/env python3
"""对随案例提供的 WikiText-2 raw 小切片执行清洗、去重和数据量统计。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

DATASET = "Salesforce/wikitext"
CONFIG = "wikitext-2-raw-v1"
SPLIT = "train"
DATASET_CARD = "https://huggingface.co/datasets/Salesforce/wikitext"
DEFAULT_INPUT = Path(__file__).resolve().parent / "data" / "wikitext2_train_sample.jsonl"
SAMPLE_OFFSETS = [0, 9000, 18000, 27000, 36000]
ROWS_PER_OFFSET = 100
MIN_WORDS = 5
MIN_ALNUM_CHARS = 40
HEADING_PATTERN = re.compile(r"^=+\s*.+?\s*=+$")


def load_rows(path: Path) -> list[str]:
    """读取 JSONL 中的 text 字段。"""
    rows: list[str] = []
    with path.open("r", encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number} 不是有效 JSONL：{error}") from error
            text = record.get("text")
            if not isinstance(text, str):
                raise ValueError(f"{path}:{line_number} 缺少字符串字段 text。")
            rows.append(text)
    if not rows:
        raise ValueError(f"输入文件没有数据行：{path}")
    return rows


def normalize_rows(rows: list[str]) -> list[str]:
    """合并空白并移除空行。"""
    return [" ".join(text.split()) for text in rows if text and text.split()]


def deduplicate_rows(rows: list[str]) -> tuple[list[str], int]:
    """对空白规范化后的全文做大小写不敏感比对，保留首次出现的行。"""
    unique: list[str] = []
    seen: set[str] = set()
    removed = 0
    for text in rows:
        key = text.casefold()
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        unique.append(text)
    return unique, removed


def passes_quality_filter(text: str) -> bool:
    """移除标题、过短片段和有效字母数字字符不足的片段。"""
    if HEADING_PATTERN.fullmatch(text):
        return False
    if len(text.split()) < MIN_WORDS:
        return False
    return sum(character.isalnum() for character in text) >= MIN_ALNUM_CHARS


def measure(rows: list[str]) -> dict[str, int]:
    return {
        "records": len(rows),
        "characters": sum(len(text) for text in rows),
        "utf8_bytes": sum(len(text.encode("utf-8")) for text in rows),
        "space_delimited_tokens": sum(len(text.split()) for text in rows),
    }


def make_stage(name: str, before: list[str], after: list[str]) -> dict[str, Any]:
    before_measure = measure(before)
    after_measure = measure(after)
    return {
        "name": name,
        "before": before_measure,
        "after": after_measure,
        "removed_records": before_measure["records"] - after_measure["records"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT, help="输入的 WikiText JSONL 样本")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="结果输出目录")
    args = parser.parse_args()

    raw_rows = load_rows(args.input_jsonl)
    normalized_rows = normalize_rows(raw_rows)
    deduplicated_rows, duplicate_count = deduplicate_rows(normalized_rows)
    cleaned_rows = [text for text in deduplicated_rows if passes_quality_filter(text)]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cleaned_path = args.output_dir / "cleaned.jsonl"
    with cleaned_path.open("w", encoding="utf-8", newline="\n") as output:
        for text in cleaned_rows:
            output.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    is_reference_sample = args.input_jsonl.resolve() == DEFAULT_INPUT.resolve()
    source_info: dict[str, Any] = {
        "dataset": DATASET if is_reference_sample else None,
        "config": CONFIG if is_reference_sample else None,
        "split": SPLIT if is_reference_sample else None,
        "dataset_card": DATASET_CARD if is_reference_sample else None,
        "sample_file": args.input_jsonl.name,
        "sample_rows": len(raw_rows),
        "sample_sha256": hashlib.sha256(args.input_jsonl.read_bytes()).hexdigest(),
    }
    if is_reference_sample:
        source_info["sample_offsets"] = SAMPLE_OFFSETS
        source_info["rows_per_offset"] = ROWS_PER_OFFSET

    summary = {
        "source": source_info,
        "rules": {
            "normalization": "合并连续空白，并移除空行",
            "deduplication": "空白规范化后全文 casefold 比对，保留首次出现的行",
            "quality_filter": {
                "remove_wiki_headings": True,
                "minimum_space_delimited_tokens": MIN_WORDS,
                "minimum_alphanumeric_characters": MIN_ALNUM_CHARS,
            },
        },
        "stages": [
            make_stage("normalize_and_remove_empty", raw_rows, normalized_rows),
            make_stage("case_insensitive_exact_deduplication", normalized_rows, deduplicated_rows),
            make_stage("quality_filter", deduplicated_rows, cleaned_rows),
        ],
        "duplicates_removed": duplicate_count,
        "cleaned_output": {
            "file": cleaned_path.name,
            **measure(cleaned_rows),
            "file_bytes": cleaned_path.stat().st_size,
        },
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

