"""A small, reproducible English/Chinese Common Crawl WET pipeline.

The pipeline follows the assignment 4 data path:

WET conversion records -> normalization -> language-specific quality filter ->
PII masking -> exact document deduplication -> MinHash/LSH near-duplicate
deduplication.

The ``en`` profile reuses assignment 4's Gopher rules.  The ``zh`` profile
implements the dependency-free rule subset documented by ChineseWebText:
minimum length, average non-empty line length, CJK character ratio, and
within-document character n-gram repetition.  It also applies the bundled,
auditable sensitive-word guard before deduplication.

It intentionally processes a bounded number of records so that learners can run
the complete flow on a laptop before scaling it to the full assignment data.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import time
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from warcio.archiveiterator import ArchiveIterator

from cs336_data.deduplication import minhash_deduplicate
from cs336_data.filter import (
    run_gopher_quality_filter,
    run_mask_emails,
    run_mask_ips,
    run_mask_phone_numbers,
)

_SPACE_RE = re.compile(r"[ \t\r\f\v]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
DEFAULT_CHINESE_SENSITIVE_WORDS_PATH = Path(__file__).with_name("zh_sensitive_words.csv")


@dataclass(frozen=True)
class PipelineConfig:
    max_records: int = 1000
    language: str = "en"
    min_chars: int = 50
    max_chars: int = 100_000
    chinese_min_chars: int = 200
    chinese_min_line_chars: int = 10
    chinese_min_cjk_ratio: float = 0.3
    chinese_ngram_size: int = 13
    chinese_max_internal_repeat: float = 0.5
    chinese_sensitive_words_path: str | None = None
    chinese_max_sensitive_words_per_line: float = 0.5
    chinese_min_sensitive_terms: int = 1
    num_hashes: int = 64
    num_bands: int = 16
    shingle_size: int = 5
    jaccard_threshold: float = 0.85


def normalize_text(text: str) -> str:
    """Normalize WET text while keeping paragraph boundaries."""

    lines = [_SPACE_RE.sub(" ", line).strip() for line in text.splitlines()]
    normalized = "\n".join(lines)
    return _BLANK_LINES_RE.sub("\n\n", normalized).strip()


def _internal_ngram_repeat_ratio(text: str, ngram_size: int) -> float:
    """Return the fraction of character n-grams occurring more than once."""

    compact = re.sub(r"\s+", "", text.casefold())
    if len(compact) < ngram_size:
        return 0.0
    ngrams = [compact[index : index + ngram_size] for index in range(len(compact) - ngram_size + 1)]
    counts = Counter(ngrams)
    repeated = sum(count for count in counts.values() if count > 1)
    return repeated / len(ngrams)


def load_sensitive_words(path: Path) -> tuple[str, ...]:
    """Load one sensitive word per CSV row, keeping the source file auditable."""

    with path.open(encoding="utf-8", newline="") as words_file:
        words = {
            row[0].strip()
            for row in csv.reader(words_file)
            if row and row[0].strip() and not row[0].lstrip().startswith("#")
        }
    return tuple(sorted(words, key=lambda word: (-len(word), word)))


def _compile_sensitive_words(words: Sequence[str]) -> re.Pattern[str] | None:
    if not words:
        return None
    return re.compile("|".join(re.escape(word) for word in words), re.IGNORECASE)


def _validate_chinese_quality_parameters(
    *,
    min_chars: int,
    min_line_chars: int,
    min_cjk_ratio: float,
    ngram_size: int,
    max_internal_repeat: float,
    max_sensitive_words_per_line: float,
    min_sensitive_terms: int,
) -> None:
    if min_chars <= 0 or min_line_chars <= 0:
        raise ValueError("Chinese length thresholds must be positive")
    if not 0.0 <= min_cjk_ratio <= 1.0:
        raise ValueError("min_cjk_ratio must be between 0 and 1")
    if ngram_size <= 0:
        raise ValueError("ngram_size must be positive")
    if not 0.0 <= max_internal_repeat <= 1.0:
        raise ValueError("max_internal_repeat must be between 0 and 1")
    if max_sensitive_words_per_line < 0.0:
        raise ValueError("max_sensitive_words_per_line must be non-negative")
    if min_sensitive_terms <= 0:
        raise ValueError("min_sensitive_terms must be positive")


def _chinese_quality_reason(
    text: str,
    *,
    min_chars: int,
    min_line_chars: int,
    min_cjk_ratio: float,
    ngram_size: int,
    max_internal_repeat: float,
    sensitive_pattern: re.Pattern[str] | None = None,
    max_sensitive_words_per_line: float = 0.5,
    min_sensitive_terms: int = 1,
) -> str | None:
    """Return the first failed ChineseWebText-style or sensitive-content rule."""

    if len(text) < min_chars:
        return "chinese_too_short"

    lines = [line for line in text.splitlines() if line.strip()]
    average_line_length = sum(map(len, lines)) / len(lines) if lines else 0.0
    if average_line_length < min_line_chars:
        return "chinese_short_lines"

    cjk_ratio = len(_CJK_RE.findall(text)) / len(text)
    if cjk_ratio < min_cjk_ratio:
        return "chinese_low_cjk_ratio"

    if sensitive_pattern is not None and lines:
        sensitive_matches = sensitive_pattern.findall(text)
        distinct_sensitive_terms = set(sensitive_matches)
        sensitive_per_line = len(distinct_sensitive_terms) / len(lines)
        if (
            len(distinct_sensitive_terms) >= min_sensitive_terms
            or sensitive_per_line > max_sensitive_words_per_line
        ):
            return "chinese_sensitive_content"

    if _internal_ngram_repeat_ratio(text, ngram_size) > max_internal_repeat:
        return "chinese_internal_repetition"
    return None


def run_chinese_quality_filter(
    text: str,
    *,
    min_chars: int = 200,
    min_line_chars: int = 10,
    min_cjk_ratio: float = 0.3,
    ngram_size: int = 13,
    max_internal_repeat: float = 0.5,
    sensitive_words: Sequence[str] | None = None,
    max_sensitive_words_per_line: float = 0.5,
    min_sensitive_terms: int = 1,
) -> bool:
    """Apply ChineseWebText heuristics plus the optional sensitive-word guard."""

    if sensitive_words is None:
        sensitive_words = load_sensitive_words(DEFAULT_CHINESE_SENSITIVE_WORDS_PATH)
    sensitive_pattern = _compile_sensitive_words(sensitive_words)

    _validate_chinese_quality_parameters(
        min_chars=min_chars,
        min_line_chars=min_line_chars,
        min_cjk_ratio=min_cjk_ratio,
        ngram_size=ngram_size,
        max_internal_repeat=max_internal_repeat,
        max_sensitive_words_per_line=max_sensitive_words_per_line,
        min_sensitive_terms=min_sensitive_terms,
    )
    return (
        _chinese_quality_reason(
            text,
            min_chars=min_chars,
            min_line_chars=min_line_chars,
            min_cjk_ratio=min_cjk_ratio,
            ngram_size=ngram_size,
            max_internal_repeat=max_internal_repeat,
            sensitive_pattern=sensitive_pattern,
            max_sensitive_words_per_line=max_sensitive_words_per_line,
            min_sensitive_terms=min_sensitive_terms,
        )
        is None
    )


def iter_wet_records(path: Path, max_records: int | None = None) -> Iterator[tuple[str, str]]:
    """Yield ``(url, text)`` from WET ``conversion`` records in source order."""

    yielded = 0
    with gzip.open(path, "rb") as stream:
        for record in ArchiveIterator(stream):
            if record.rec_type != "conversion":
                continue
            payload = record.content_stream().read()
            url = record.rec_headers.get_header("WARC-Target-URI") or ""
            yield url, payload.decode("utf-8", errors="replace")
            yielded += 1
            if max_records is not None and yielded >= max_records:
                return


def mask_pii(text: str) -> tuple[str, Counter[str]]:
    """Apply the PII masking helpers already used by assignment 4."""

    counts: Counter[str] = Counter()
    text, count = run_mask_emails(text)
    counts["emails"] += count
    text, count = run_mask_phone_numbers(text)
    counts["phones"] += count
    text, count = run_mask_ips(text)
    counts["ips"] += count
    return text, counts


def run_pipeline(input_path: Path, output_dir: Path, config: PipelineConfig) -> dict:
    """Run the bounded pipeline and write ``filtered.jsonl`` and ``stats.json``."""

    if config.language not in {"en", "zh"}:
        raise ValueError(f"unsupported language profile: {config.language}")
    if config.max_records <= 0:
        raise ValueError("max_records must be positive")
    sensitive_pattern = None
    sensitive_words_path: Path | None = None
    if config.language == "zh":
        sensitive_words_path = (
            Path(config.chinese_sensitive_words_path)
            if config.chinese_sensitive_words_path
            else DEFAULT_CHINESE_SENSITIVE_WORDS_PATH
        )
        sensitive_pattern = _compile_sensitive_words(load_sensitive_words(sensitive_words_path))
    _validate_chinese_quality_parameters(
        min_chars=config.chinese_min_chars,
        min_line_chars=config.chinese_min_line_chars,
        min_cjk_ratio=config.chinese_min_cjk_ratio,
        ngram_size=config.chinese_ngram_size,
        max_internal_repeat=config.chinese_max_internal_repeat,
        max_sensitive_words_per_line=config.chinese_max_sensitive_words_per_line,
        min_sensitive_terms=config.chinese_min_sensitive_terms,
    )
    minimum_chars = config.chinese_min_chars if config.language == "zh" else config.min_chars
    if config.min_chars <= 0 or config.max_chars < minimum_chars:
        raise ValueError("require the selected minimum length to be no greater than max_chars")
    started = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    stats: Counter[str] = Counter(
        {
            key: 0
            for key in (
                "input_records",
                "empty",
                "too_short",
                "too_long",
                "gopher_rejected",
                "chinese_too_short",
                "chinese_short_lines",
                "chinese_low_cjk_ratio",
                "chinese_sensitive_content",
                "chinese_internal_repetition",
                "chinese_quality_rejected",
                "kept_before_dedup",
                "exact_duplicates",
                "after_exact_dedup",
                "minhash_candidate_pairs",
                "minhash_duplicates",
                "output_records",
            )
        }
    )
    pii_counts: Counter[str] = Counter()
    filtered: list[dict[str, str]] = []

    for url, raw_text in iter_wet_records(input_path, config.max_records):
        stats["input_records"] += 1
        text = normalize_text(raw_text)
        if not text:
            stats["empty"] += 1
            continue
        if len(text) > config.max_chars:
            stats["too_long"] += 1
            continue
        if config.language == "en":
            if len(text) < config.min_chars:
                stats["too_short"] += 1
                continue
            if not run_gopher_quality_filter(text):
                stats["gopher_rejected"] += 1
                continue
        elif config.language == "zh":
            reason = _chinese_quality_reason(
                text,
                min_chars=config.chinese_min_chars,
                min_line_chars=config.chinese_min_line_chars,
                min_cjk_ratio=config.chinese_min_cjk_ratio,
                ngram_size=config.chinese_ngram_size,
                max_internal_repeat=config.chinese_max_internal_repeat,
                sensitive_pattern=sensitive_pattern,
                max_sensitive_words_per_line=config.chinese_max_sensitive_words_per_line,
                min_sensitive_terms=config.chinese_min_sensitive_terms,
            )
            if reason is not None:
                stats[reason] += 1
                stats["chinese_quality_rejected"] += 1
                continue

        text, current_pii_counts = mask_pii(text)
        pii_counts.update(current_pii_counts)
        filtered.append(
            {
                "url": url,
                "text": text,
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
        )

    stats["kept_before_dedup"] = len(filtered)

    exact_kept: list[dict[str, str]] = []
    seen_hashes: set[str] = set()
    for document in filtered:
        digest = document["text_sha256"]
        if digest in seen_hashes:
            stats["exact_duplicates"] += 1
            continue
        seen_hashes.add(digest)
        exact_kept.append(document)

    stats["after_exact_dedup"] = len(exact_kept)
    kept_indices, minhash_stats = minhash_deduplicate(
        [document["text"] for document in exact_kept],
        num_hashes=config.num_hashes,
        num_bands=config.num_bands,
        shingle_size=config.shingle_size,
        jaccard_threshold=config.jaccard_threshold,
    )
    stats["minhash_candidate_pairs"] = minhash_stats.candidate_pairs
    stats["minhash_duplicates"] = minhash_stats.duplicate_documents
    final_documents = [exact_kept[index] for index in kept_indices]
    stats["output_records"] = len(final_documents)

    output_path = output_dir / "filtered.jsonl"
    with output_path.open("w", encoding="utf-8") as output_file:
        for document in final_documents:
            output_file.write(json.dumps(document, ensure_ascii=False) + "\n")

    stats_payload = {
        "input_path": str(input_path),
        "config": asdict(config),
        "resolved_chinese_sensitive_words_path": str(sensitive_words_path) if sensitive_words_path else None,
        "counts": dict(stats),
        "pii_replacements": dict(pii_counts),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    with (output_dir / "stats.json").open("w", encoding="utf-8") as stats_file:
        json.dump(stats_payload, stats_file, ensure_ascii=False, indent=2)

    for key, value in stats.items():
        print(f"{key}={value}")
    print(f"output={output_path}")
    print(f"stats={output_dir / 'stats.json'}")
    return stats_payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="A local .warc.wet.gz file")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=1000)
    parser.add_argument("--language", choices=("en", "zh"), default="en")
    parser.add_argument("--min-chars", type=int, default=50)
    parser.add_argument("--max-chars", type=int, default=100_000)
    parser.add_argument("--chinese-min-chars", type=int, default=200)
    parser.add_argument("--chinese-min-line-chars", type=int, default=10)
    parser.add_argument("--chinese-min-cjk-ratio", type=float, default=0.3)
    parser.add_argument("--chinese-ngram-size", type=int, default=13)
    parser.add_argument("--chinese-max-internal-repeat", type=float, default=0.5)
    parser.add_argument("--chinese-sensitive-words", type=Path, default=None)
    parser.add_argument("--chinese-max-sensitive-words-per-line", type=float, default=0.5)
    parser.add_argument("--chinese-min-sensitive-terms", type=int, default=1)
    parser.add_argument("--num-hashes", type=int, default=64)
    parser.add_argument("--num-bands", type=int, default=16)
    parser.add_argument("--shingle-size", type=int, default=5)
    parser.add_argument("--jaccard-threshold", type=float, default=0.85)
    args = parser.parse_args()
    if args.max_records <= 0:
        parser.error("--max-records must be positive")
    if args.chinese_min_chars <= 0 or args.chinese_min_line_chars <= 0:
        parser.error("Chinese length thresholds must be positive")
    if not 0.0 <= args.chinese_min_cjk_ratio <= 1.0:
        parser.error("--chinese-min-cjk-ratio must be between 0 and 1")
    if args.chinese_ngram_size <= 0:
        parser.error("--chinese-ngram-size must be positive")
    if not 0.0 <= args.chinese_max_internal_repeat <= 1.0:
        parser.error("--chinese-max-internal-repeat must be between 0 and 1")
    if args.chinese_max_sensitive_words_per_line < 0.0:
        parser.error("--chinese-max-sensitive-words-per-line must be non-negative")
    if args.chinese_min_sensitive_terms <= 0:
        parser.error("--chinese-min-sensitive-terms must be positive")
    selected_min_chars = args.chinese_min_chars if args.language == "zh" else args.min_chars
    if args.min_chars <= 0 or args.max_chars < selected_min_chars:
        parser.error("the selected minimum length must be no greater than --max-chars")
    return args


def main() -> None:
    args = _parse_args()
    config = PipelineConfig(
        max_records=args.max_records,
        language=args.language,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        chinese_min_chars=args.chinese_min_chars,
        chinese_min_line_chars=args.chinese_min_line_chars,
        chinese_min_cjk_ratio=args.chinese_min_cjk_ratio,
        chinese_ngram_size=args.chinese_ngram_size,
        chinese_max_internal_repeat=args.chinese_max_internal_repeat,
        chinese_sensitive_words_path=(str(args.chinese_sensitive_words) if args.chinese_sensitive_words else None),
        chinese_max_sensitive_words_per_line=args.chinese_max_sensitive_words_per_line,
        chinese_min_sensitive_terms=args.chinese_min_sensitive_terms,
        num_hashes=args.num_hashes,
        num_bands=args.num_bands,
        shingle_size=args.shingle_size,
        jaccard_threshold=args.jaccard_threshold,
    )
    run_pipeline(args.input, args.output_dir, config)


if __name__ == "__main__":
    main()
