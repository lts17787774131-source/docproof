"""端到端冒烟测试：解析 → 切分 → 检索 → 拒答/回答 → 指标。

跑法：.venv312\\Scripts\\python.exe -m pytest tests -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from docproof.chunking.structural import build_chunks  # noqa: E402
from docproof.config import SETTINGS  # noqa: E402
from docproof.eval.metrics import answer_match, evidence_in  # noqa: E402
from docproof.generate.answer import Answerer  # noqa: E402
from docproof.generate.providers import ExtractiveProvider  # noqa: E402
from docproof.index import store  # noqa: E402
from docproof.index.bm25 import BM25  # noqa: E402
from docproof.ingest.pipeline import parse_pdf  # noqa: E402
from docproof.ir import Chunk, Document  # noqa: E402
from docproof.retrieve.hybrid import Retriever  # noqa: E402

PDF = ROOT / "data" / "raw" / "华信科技2023年年度报告.pdf"


@pytest.fixture(scope="module")
def doc():
    if not PDF.exists():
        pytest.skip("corpus missing, run scripts/make_corpus.py first")
    return parse_pdf(PDF, SETTINGS)


@pytest.fixture(scope="module")
def chunks(doc):
    return build_chunks(doc, mode="structural", settings=SETTINGS, table_view="rows")


def test_parse_has_tables_and_figures(doc):
    assert doc.meta["n_pages"] >= 6
    assert doc.meta["n_tables"] >= 4
    assert doc.meta["n_figures"] >= 1
    types = {b.type for b in doc.blocks}
    assert {"heading", "text", "table", "figure"} <= types


def test_sections_are_hierarchical(doc):
    sections = doc.sections()
    assert any("第一节" in s for s in sections)
    assert any(">" in s for s in sections)  # 出现过二级章节


def test_table_html_and_bbox(doc):
    tables = [b for b in doc.blocks if b.type == "table"]
    t = tables[0]
    assert "<table>" in t.html and "<th>" in t.html
    assert t.cells and all(len(c["bbox"]) == 4 for c in t.cells)
    # 行式序列化里，每行都自带列名
    assert "=" in chunks_text(t)


def chunks_text(block) -> str:
    from docproof.ir import table_rows_text
    return table_rows_text(block.html)


def test_chunks_have_parent_and_page(chunks):
    assert len(chunks) >= 15
    for c in chunks:
        assert c.page >= 1
        assert c.parent_text
        assert c.bbox[2] > c.bbox[0]


def test_cross_page_table_merged(chunks):
    merged = [c for c in chunks if "续上页" in c.text and c.type == "table"]
    assert merged, "跨页续表应被并回上一张表"
    assert "2023" in merged[0].text


def test_bm25_ranks_relevant_first():
    docs = ["公司聘请立信会计师事务所", "研发投入合计 4.82 亿元", "董事会秘书变更"]
    bm = BM25(docs)
    assert bm.search("会计师事务所", 3)[0][0] == 0


def test_answer_and_refusal(chunks):
    retriever = Retriever(chunks)
    answerer = Answerer(retriever, ExtractiveProvider(), SETTINGS)

    ok = answerer.ask("第三季度营业收入是多少亿元？", doc_id="t")
    assert not ok.refused
    assert "12.4" in ok.answer
    assert ok.citations and ok.citations[0].page == 2

    bad = answerer.ask("公司 2025 年的营业收入是多少？", doc_id="t")
    assert bad.refused and bad.refusal_reason == "number_absent"


def test_compare_operator(chunks):
    retriever = Retriever(chunks)
    answerer = Answerer(retriever, ExtractiveProvider(), SETTINGS)
    ans = answerer.ask("哪一个季度的营业收入最高？", doc_id="t")
    assert "第四季度" in ans.answer


def test_arith_operator(chunks):
    """计算题：12.4 − 12.4 = 0.0，且引用校验看的是"依据的数字"。"""
    retriever = Retriever(chunks)
    answerer = Answerer(retriever, ExtractiveProvider(), SETTINGS)
    ans = answerer.ask("第三季度营业收入相比上年同期增长了多少？", doc_id="t")
    assert "0.0" in ans.answer
    assert ans.citations and ans.citations[0].verified


def test_multi_value_operator(chunks):
    """多值题：必须把 2022 和 2023 两行都拼进来。"""
    retriever = Retriever(chunks)
    answerer = Answerer(retriever, ExtractiveProvider(), SETTINGS)
    ans = answerer.ask("经营活动产生的现金流量净额最近两年分别是多少？", doc_id="t")
    assert "5.6" in ans.answer and "7.3" in ans.answer


def test_metrics_helpers():
    assert answer_match("12.4 亿元", "12.4 亿元")
    assert not answer_match("11.6", "12.4")
    assert evidence_in("第三季度：营业收入（亿元）=12.4", "第三季度 12.4")


def test_index_delete_keeps_document_then_purge(tmp_path):
    """删索引必须保留 documents 行（重建要 source_path），purge 才删干净。"""
    conn = store.connect(tmp_path / "t.db")
    doc = Document(doc_id="d1", title="t", source_path="x.pdf",
                   page_sizes=[[1.0, 1.0]], blocks=[])
    store.upsert_document(conn, doc)
    store.replace_chunks(conn, "d1", [
        Chunk(id="d1:c0", doc_id="d1", idx=0, page=1, type="text", section="", text="hello")])
    assert len(store.fetch_chunks(conn, "d1")) == 1

    assert store.delete_index(conn, "d1") == 1
    assert store.fetch_chunks(conn, "d1") == []
    assert store.get_document(conn, "d1") is not None      # 记录还在，才能重建

    store.purge_document(conn, "d1")
    assert store.get_document(conn, "d1") is None
    assert not evidence_in("研发投入 4.82 亿元", "毛利率 34.1%")
