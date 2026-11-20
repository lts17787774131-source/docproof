"""真实年报口语问题压测：检验算子在"非模板提问"上还剩多少。

这份集合只有 15 题，作用是**暴露失败模式**，不是刷分：
- 同一份真实年报（广咨国际 2025，161 页），问题按真实用户的说法写（"多少""哪一家""去年"）；
- 答案从解析出的表格/正文里人工核对过，写在 real_qa.jsonl；
- 跑完直接看错题——比看总分有用。

用法：python scripts/eval_real.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from docproof.config import SETTINGS  # noqa: E402
from docproof.eval.metrics import answer_match  # noqa: E402
from docproof.generate.answer import Answerer, doc_year_of  # noqa: E402
from docproof.generate.providers import get_provider  # noqa: E402
from docproof.index import store  # noqa: E402
from docproof.index.dense import OllamaEmbedder  # noqa: E402
from docproof.retrieve.hybrid import Retriever  # noqa: E402


def build(doc_id: str, provider: str = "extractive") -> Answerer:
    conn = store.connect(SETTINGS.db_path)
    chunks = store.fetch_chunks(conn, doc_id)
    if not chunks:
        raise SystemExit(f"doc not indexed: {doc_id}")
    _, mat = store.load_embeddings(conn, doc_id, SETTINGS.embed_model)
    embedder = OllamaEmbedder(SETTINGS.ollama_url, SETTINGS.embed_model,
                              cache_path=SETTINGS.data_dir / "embed_cache.db")
    retriever = Retriever(chunks, embedder=embedder,
                          embeddings=([c.id for c in chunks], mat) if mat is not None else None)
    return Answerer(retriever, get_provider(provider, SETTINGS), SETTINGS)


def main() -> None:
    items = [json.loads(line) for line in
             (ROOT / "eval" / "datasets" / "real_qa.jsonl").read_text(encoding="utf-8").splitlines()
             if line.strip()]
    answerers: dict[str, Answerer] = {}
    years: dict[str, int | None] = {}
    conn = store.connect(SETTINGS.db_path)
    chunk_cache: dict[str, list] = {}
    ok = 0
    rows = []
    for it in items:
        did = it["doc_id"]
        if did not in answerers:
            answerers[did] = build(did)
            chunk_cache[did] = store.fetch_chunks(conn, did)
            info = store.get_document(conn, did) or {}
            years[did] = doc_year_of(info.get("meta"), info.get("title", ""), chunk_cache[did])
        ans = answerers[did].ask(it["question"], doc_id=did, doc_year=years[did])
        if it["qtype"] == "T6":
            correct = ans.refused
        else:
            correct = (not ans.refused) and answer_match(ans.answer, it["answer"])
        ok += int(correct)
        rows.append((correct, it, ans))
        flag = "PASS" if correct else "FAIL"
        print(f"[{flag}] {it['qid']} {it['question']}")
        print(f"        gold={it['answer']!r}")
        print(f"        pred={ans.answer[:90]!r} refused={ans.refused} "
              f"[{ans.refusal_reason}] cite={[c.page for c in ans.citations]}")

    acc = ok / len(items)
    print(f"\n真实口语集：{ok}/{len(items)} = {acc*100:.0f}%")
    fails = [r for r in rows if not r[0]]
    if fails:
        print("失败模式：")
        for correct, it, ans in fails:
            print(f"  - {it['qtype']} {it['question']} -> {ans.refusal_reason or ans.answer[:40]}")


if __name__ == "__main__":
    main()
