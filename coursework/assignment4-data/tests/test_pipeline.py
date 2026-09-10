from pathlib import Path

import pytest

from cs336_data.deduplication import minhash_deduplicate
from cs336_data.pipeline import (
    PipelineConfig,
    normalize_text,
    run_chinese_quality_filter,
    run_pipeline,
)

CHINESE_TEXT = """数据工程需要先读取网页归档中的纯文本记录，并保存来源网址。
长度检查可以移除缺少上下文的短片段，也能发现异常膨胀的页面。
中文字符比例用于排除导航、脚本和其他非中文内容。
文档内部重复率能够识别同一段广告或模板被反复拼接的情况。
精确去重先比较稳定摘要，再把计算成本交给近重复检索。
MinHash 只生成候选文档对，最终结果还要经过 Jaccard 相似度复核。
输出文件采用 JSONL 格式，每条记录包含网址、文本和文本摘要。
统计文件保存输入量、过滤量以及两次去重后的剩余数量。
固定数据地址和文件校验值让其他人可以复现实验结果。
示例限制处理条数，因此普通 CPU 也能在较短时间内跑完整条流程。
如果调整阈值，应重新查看被过滤样本，避免误删正常内容。
中文规则只负责启发式清洗，不替代人工抽样或专门的质量模型。"""


def test_normalize_text_keeps_paragraph_boundaries():
    assert normalize_text(" first  line\n\n\n second\tline ") == "first line\n\nsecond line"


def test_minhash_is_deterministic_and_keeps_first_document():
    texts = [
        "A stable document about language modeling and data quality.",
        "A stable document about language modeling and data cleaning.",
        "An unrelated document about marine biology and whales.",
    ]
    first, first_stats = minhash_deduplicate(texts, num_hashes=32, num_bands=8, shingle_size=5, jaccard_threshold=0.65)
    second, second_stats = minhash_deduplicate(
        texts, num_hashes=32, num_bands=8, shingle_size=5, jaccard_threshold=0.65
    )
    assert first == second
    assert first_stats == second_stats
    assert first[0] == 0


def test_minhash_validates_parameters_without_documents():
    with pytest.raises(ValueError, match="divisible"):
        minhash_deduplicate([], num_hashes=3, num_bands=2)


def test_chinese_quality_filter_uses_documented_rules():
    assert run_chinese_quality_filter(CHINESE_TEXT)
    assert not run_chinese_quality_filter("This is an English document. " * 20)
    assert not run_chinese_quality_filter("这是一个重复段落，用于检查文档内部重复率。\n" * 20)
    sensitive_text = CHINESE_TEXT + "\n" + "风险词 " * 10
    assert not run_chinese_quality_filter(sensitive_text, sensitive_words=("风险词",))
    assert not run_chinese_quality_filter(CHINESE_TEXT + "\n风险词", sensitive_words=("风险词",))


def test_pipeline_reports_each_stage(monkeypatch, tmp_path: Path):
    repeated = (
        "This is a useful English document for testing a reproducible data pipeline. "
        "It contains enough natural language to pass the Gopher quality checks. "
        "Contact test@example.com or call 283-182-3829; server 192.0.2.1. "
    ) * 8
    near_duplicate = repeated.replace("useful", "practical")
    records = iter(
        [
            ("https://example.test/first", repeated),
            ("https://example.test/exact", repeated),
            ("https://example.test/near", near_duplicate),
            ("https://example.test/short", "too short"),
        ]
    )
    monkeypatch.setattr("cs336_data.pipeline.iter_wet_records", lambda *_args: records)

    result = run_pipeline(
        Path("unused.warc.wet.gz"),
        tmp_path,
        PipelineConfig(num_hashes=32, num_bands=8, jaccard_threshold=0.65),
    )

    counts = result["counts"]
    assert counts["input_records"] == 4
    assert counts["too_short"] == 1
    assert counts["kept_before_dedup"] == 3
    assert counts["exact_duplicates"] == 1
    assert counts["minhash_duplicates"] == 1
    assert counts["output_records"] == 1
    assert result["pii_replacements"] == {"emails": 24, "phones": 24, "ips": 24}
    output = (tmp_path / "filtered.jsonl").read_text(encoding="utf-8")
    assert "test@example.com" not in output
    assert "|||EMAIL_ADDRESS|||" in output
    assert output.count("\n") == 1


def test_pipeline_reports_chinese_quality_stage(monkeypatch, tmp_path: Path):
    records = iter(
        [
            ("https://example.test/zh-first", CHINESE_TEXT),
            ("https://example.test/zh-exact", CHINESE_TEXT),
            ("https://example.test/zh-near", CHINESE_TEXT.replace("固定数据地址", "稳定数据地址")),
            ("https://example.test/en", "This is an English document. " * 20),
        ]
    )
    monkeypatch.setattr("cs336_data.pipeline.iter_wet_records", lambda *_args: records)

    result = run_pipeline(
        Path("unused.warc.wet.gz"),
        tmp_path,
        PipelineConfig(
            language="zh",
            chinese_min_chars=200,
            num_hashes=32,
            num_bands=8,
            jaccard_threshold=0.65,
        ),
    )

    counts = result["counts"]
    assert counts["input_records"] == 4
    assert counts["chinese_low_cjk_ratio"] == 1
    assert counts["chinese_quality_rejected"] == 1
    assert counts["kept_before_dedup"] == 3
    assert counts["exact_duplicates"] == 1
    assert counts["minhash_duplicates"] == 1
    assert counts["output_records"] == 1


def test_pipeline_reports_chinese_sensitive_stage(monkeypatch, tmp_path: Path):
    sensitive_words = tmp_path / "sensitive.csv"
    sensitive_words.write_text("风险词,\n", encoding="utf-8")
    records = iter(
        [
            ("https://example.test/zh", CHINESE_TEXT),
            ("https://example.test/zh-sensitive", CHINESE_TEXT + "\n" + "风险词 " * 10),
        ]
    )
    monkeypatch.setattr("cs336_data.pipeline.iter_wet_records", lambda *_args: records)

    result = run_pipeline(
        Path("unused.warc.wet.gz"),
        tmp_path / "output",
        PipelineConfig(
            language="zh",
            chinese_sensitive_words_path=str(sensitive_words),
        ),
    )

    counts = result["counts"]
    assert counts["input_records"] == 2
    assert counts["chinese_sensitive_content"] == 1
    assert counts["chinese_quality_rejected"] == 1
    assert counts["output_records"] == 1
