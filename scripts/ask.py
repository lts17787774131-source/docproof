"""单题问答 CLI：验证检索 + 生成 + 引用。

用法：
    python scripts/ask.py "第三季度营业收入是多少" --doc-id <id> --provider extractive
    python scripts/ask.py "第三季度营业收入是多少" --provider ollama
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docproof.config import SETTINGS  # noqa: E402
from docproof.generate.answer import Answerer  # noqa: E402
from docproof.generate.providers import get_provider  # noqa: E402
from docproof.index import store  # noqa: E402
from docproof.index.dense import OllamaEmbedder  # noqa: E402
from docproof.retrieve.hybrid import Retriever  # noqa: E402


def build_answerer(doc_id: str, provider_name: str, use_dense: bool = True) -> Answerer:
    conn = store.connect(SETTINGS.db_path)
    chunks = store.fetch_chunks(conn, doc_id)
    if not chunks:
        raise SystemExit(f"doc not indexed: {doc_id}")
    embedder = OllamaEmbedder(SETTINGS.ollama_url, SETTINGS.embed_model,
                              cache_path=SETTINGS.data_dir / "embed_cache.db") if use_dense else None
    embeddings = store.load_embeddings(conn, doc_id, SETTINGS.embed_model) if use_dense else None
    retriever = Retriever(chunks, embedder=embedder, embeddings=embeddings)
    return Answerer(retriever, get_provider(provider_name, SETTINGS), SETTINGS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--doc-id", default="")
    ap.add_argument("--provider", default="extractive",
                    choices=["extractive", "ollama", "deepseek"])
    ap.add_argument("--no-dense", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    conn = store.connect(SETTINGS.db_path)
    doc_id = args.doc_id
    if not doc_id:
        docs = store.list_documents(conn)
        if not docs:
            raise SystemExit("no documents indexed; run scripts/build_index.py first")
        doc_id = docs[0]["doc_id"]

    answerer = build_answerer(doc_id, args.provider, use_dense=not args.no_dense)
    payload = answerer.ask(args.question, doc_id=doc_id)

    if args.json:
        print(json.dumps(payload.to_dict(), ensure_ascii=False, indent=2))
        return

    print(f"Q: {payload.question}")
    print(f"A: {payload.answer}   [refused={payload.refused} reason={payload.refusal_reason}]")
    print(f"provider={payload.provider} route={payload.route} {payload.latency_ms}ms")
    for c in payload.citations:
        print(f"  [{c.eid}] p{c.page} {c.type} verified={c.verified} bbox={tuple(round(v,3) for v in c.bbox)}")
        print(f"       {c.snippet[:100]}")
    print("  top hits:")
    for h in payload.hits[:5]:
        print(f"    {h['score']:.4f} p{h['page']} {h['type']:<6} {h['text'][:60]}")


if __name__ == "__main__":
    main()
