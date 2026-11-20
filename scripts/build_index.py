"""建索引：PDF → DocIR → 切分 → SQLite →（可选）向量。

用法：
    python scripts/build_index.py                  # 全部 PDF，结构切分 + 向量
    python scripts/build_index.py --mode fixed     # 基线 B0 的定长切分
    python scripts/build_index.py --no-dense       # 只建 BM25
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docproof.chunking.structural import build_chunks  # noqa: E402
from docproof.config import SETTINGS  # noqa: E402
from docproof.index import store  # noqa: E402
from docproof.index.dense import OllamaEmbedder  # noqa: E402
from docproof.ingest.pipeline import parse_pdf  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(SETTINGS.raw_dir))
    ap.add_argument("--mode", default="structural", choices=["structural", "fixed"])
    ap.add_argument("--no-dense", action="store_true")
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()

    SETTINGS.ensure_dirs()
    if args.reset and SETTINGS.db_path.exists():
        SETTINGS.db_path.unlink()

    conn = store.connect(SETTINGS.db_path)
    embedder = None if args.no_dense else OllamaEmbedder(
        SETTINGS.ollama_url, SETTINGS.embed_model, cache_path=SETTINGS.data_dir / "embed_cache.db")

    pdfs = sorted(Path(args.raw).glob("*.pdf"))
    if not pdfs:
        print("no pdf found in", args.raw)
        return

    for pdf in pdfs:
        t0 = time.time()
        doc = parse_pdf(pdf, SETTINGS)
        chunks = build_chunks(doc, mode=args.mode, settings=SETTINGS)
        store.upsert_document(conn, doc)
        store.replace_chunks(conn, doc.doc_id, chunks)

        if embedder is not None:
            vecs = embedder.embed([c.text for c in chunks])
            store.save_embeddings(conn, [c.id for c in chunks], vecs, SETTINGS.embed_model)

        print(f"[index] {pdf.name} -> {doc.doc_id} | pages={doc.meta['n_pages']} "
              f"blocks={doc.meta['n_blocks']} tables={doc.meta['n_tables']} "
              f"figs={doc.meta['n_figures']} chunks={len(chunks)} "
              f"({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
