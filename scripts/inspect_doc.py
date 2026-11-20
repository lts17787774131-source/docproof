"""调试：打印一份 PDF 的 DocIR 摘要（block 类型、表格、章节）。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docproof.config import SETTINGS  # noqa: E402
from docproof.ingest.pipeline import parse_pdf  # noqa: E402


def main(path: str) -> None:
    SETTINGS.ensure_dirs()
    doc = parse_pdf(path)
    print(f"doc_id={doc.doc_id} title={doc.title}")
    print(f"pages={doc.meta['n_pages']} blocks={doc.meta['n_blocks']} "
          f"tables={doc.meta['n_tables']} figures={doc.meta['n_figures']} "
          f"scanned={doc.meta['scanned_pages']} sec={doc.meta['parse_seconds']}s")
    print("sections:", " | ".join(doc.sections())[:300])
    for b in doc.blocks:
        txt = (b.text or "").replace("\n", " ")[:60]
        print(f"p{b.page:<2} {b.type:<8} {b.bbox[0]:.2f},{b.bbox[1]:.2f} "
              f"sec={b.section[:28]:<28} {txt}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/raw/华信科技2023年年度报告.pdf")
