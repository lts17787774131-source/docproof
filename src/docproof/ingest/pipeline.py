"""解析编排：PDF → DocIR（+ 页面图）.

一次解析产出三样东西，后面所有模块都从这里取：
1. ``Document``（blocks 带 page/bbox/section）；
2. 页面 PNG（给前端高亮与 VLM 用）；
3. 解析统计（表格数、图数、扫描页），进 trace。
"""
from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path

import pymupdf

from ..config import SETTINGS, Settings
from ..ir import Block, Document
from .pdf_text import extract_text_blocks, global_body_size, page_char_count
from .tables import attach_table_captions, extract_tables

FIG_CAP_RE = re.compile(r"^\s*(图|figure|Figure)\s*\d+")


def _center_inside(inner: tuple, outer: tuple) -> bool:
    cx = (inner[0] + inner[2]) / 2
    cy = (inner[1] + inner[3]) / 2
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def _area(b: tuple) -> float:
    return max((b[2] - b[0]) * (b[3] - b[1]), 1e-9)


def _covered_ratio(inner: tuple, outer: tuple) -> float:
    """文本块被图片覆盖的比例（交集 / 文本面积）。"""
    ix0, iy0 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix1, iy1 = min(inner[2], outer[2]), min(inner[3], outer[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    return (ix1 - ix0) * (iy1 - iy0) / _area(inner)


def _doc_id(path: Path) -> str:
    h = hashlib.sha256(path.name.encode("utf-8")).hexdigest()[:10]
    return f"{path.stem[:24]}_{h}"


def _extract_figures(page: pymupdf.Page, page_no: int, doc_id: str,
                     text_blocks: list[Block]) -> list[Block]:
    width, height = page.rect.width, page.rect.height
    figs: list[Block] = []
    data = page.get_text("dict")
    k = 0
    for block in data.get("blocks", []):
        if block.get("type") != 1:
            continue
        x0, y0, x1, y1 = block["bbox"]
        if (x1 - x0) < 40 or (y1 - y0) < 40:  # 跳过装饰性小图
            continue
        norm = (x0 / width, y0 / height, x1 / width, y1 / height)
        caption = ""
        for tb in text_blocks:
            if tb.page != page_no or not FIG_CAP_RE.match(tb.text):
                continue
            dy = tb.bbox[1] - norm[3]
            if -0.01 <= dy <= 0.08:
                caption = tb.text.strip().replace("\n", " ")
                tb.meta["attached"] = f"{doc_id}:f{page_no}_{k}"
                break
        figs.append(
            Block(
                id=f"{doc_id}:f{page_no}_{k}",
                doc_id=doc_id,
                page=page_no,
                type="figure",
                bbox=norm,
                text=caption,
                caption=caption,
                meta={"w": round(x1 - x0, 1), "h": round(y1 - y0, 1)},
            )
        )
        k += 1
    return figs


STMT_RE = re.compile(r"(合并|母公司)\s*(资产负债表|利润表|现金流量表|所有者权益变动表|股东权益变动表)")


def _tag_table_subject(blocks: list[Block]) -> None:
    """给财务报表的表格打上（主体, 报表名）标签。

    真实年报里"合并营业收入"和"母公司营业收入"是两个数，字面完全一样，
    检索层无法区分。这里用报表标题（通常在本页或上一页的页首）判断主体。
    """
    for b in blocks:
        if b.type != "table":
            continue
        best = None
        for x in blocks:
            if x is b or x.type not in ("heading", "text"):
                continue
            if x.page not in (b.page, b.page - 1):
                continue
            m = STMT_RE.search(x.text or "")
            if not m:
                continue
            # 同页要求在上方；上一页的页首标题也认
            if x.page == b.page and x.bbox[1] > b.bbox[1]:
                continue
            key = (0 if x.page == b.page else 1, abs(x.bbox[1] - b.bbox[1]))
            if best is None or key < best[0]:
                best = (key, m)
        if best:
            subj, stmt = best[1].group(1), best[1].group(2)
            b.meta["subject"] = "parent" if subj == "母公司" else "consolidated"
            b.meta["statement"] = stmt


def _assign_sections(blocks: list[Block]) -> None:
    stack: list[tuple[int, str]] = []
    current = ""
    for b in blocks:
        if b.type == "heading":
            level = int(b.meta.get("level") or 2)
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, b.text.strip().replace("\n", " ")))
            current = " > ".join(t for _, t in stack)
        b.section = current


def parse_pdf(path: str | Path, settings: Settings = SETTINGS,
              doc_id: str | None = None, title: str | None = None,
              render_pages: bool = True) -> Document:
    path = Path(path)
    t0 = time.time()
    doc = pymupdf.open(path)
    doc_id = doc_id or _doc_id(path)
    pages = list(doc)
    if settings.max_pages:
        pages = pages[: settings.max_pages]

    # 第一遍：先拿表格矩形，供全文字号统计排除
    tables_by_page: list[list[Block]] = []
    rects_by_page: list[list[tuple]] = []
    for i, page in enumerate(pages, start=1):
        tbls = extract_tables(page, i, doc_id)
        tables_by_page.append(tbls)
        w, h = page.rect.width, page.rect.height
        rects_by_page.append([(t.bbox[0] * w, t.bbox[1] * h, t.bbox[2] * w, t.bbox[3] * h)
                              for t in tbls])

    body = global_body_size(pages, rects_by_page)

    blocks: list[Block] = []
    page_sizes: list[list[float]] = []
    scanned_pages: list[int] = []
    n_tables = n_figs = 0

    for i, page in enumerate(pages, start=1):
        w, h = page.rect.width, page.rect.height
        page_sizes.append([w, h])

        texts = extract_text_blocks(page, i, doc_id, body_size=body, exclude=rects_by_page[i - 1])
        tables = tables_by_page[i - 1]
        attach_table_captions(tables, texts)

        if tables:
            texts = [tb for tb in texts
                     if not any(_center_inside(tb.bbox, t.bbox) for t in tables)]

        figs = _extract_figures(page, i, doc_id, texts)
        if figs:
            # 只丢掉"确实被图片盖住"的文本：早期用"中心点落在图片内"，
            # 遇到整页背景图的设计型简历会把所有文字删光（实测 1096 字 → 0 字）
            page_area = w * h
            real_figs = [f for f in figs if _area(f.bbox) * page_area < 0.5 * page_area]
            texts = [t for t in texts
                     if not any(_covered_ratio(t.bbox, f.bbox) > 0.9 for f in real_figs)]

        if page_char_count(page) < settings.text_layer_min_chars and not tables:
            scanned_pages.append(i)

        blocks.extend(texts)
        blocks.extend(tables)
        blocks.extend(figs)
        n_tables += len(tables)
        n_figs += len(figs)

        if render_pages:
            out = settings.pages_dir / doc_id / f"p{i}.png"
            if not out.exists():
                out.parent.mkdir(parents=True, exist_ok=True)
                page.get_pixmap(dpi=settings.page_dpi).save(out)

    # 已被表格/图片"吸附"的 caption 块不再单独成块，避免重复证据
    blocks = [b for b in blocks if not b.meta.get("attached")]
    blocks.sort(key=lambda b: (b.page, round(b.bbox[1], 3), b.bbox[0]))
    for idx, b in enumerate(blocks):
        b.order = idx
    _assign_sections(blocks)
    _tag_table_subject(blocks)

    if title is None:
        if len(page_sizes) <= 1:
            # 单页文档（简历/JD/通知）通常没有"大标题"，
            # 取第一个大字号块会拿到项目名，直接用文件名更准
            title = path.stem
        else:
            heads = [b for b in blocks if b.type == "heading" and b.page == 1 and b.bbox[1] < 0.35]
            heads.sort(key=lambda b: (-float(b.meta.get("font_size") or 0), b.bbox[1]))
            title = heads[0].text.strip().replace("\n", " ") if heads else path.stem

    m_year = re.search(r"(20\d{2})\s*年", title) or re.search(r"(20\d{2})\s*年", blocks[0].text if blocks else "")
    report_year = int(m_year.group(1)) if m_year else None

    document = Document(
        doc_id=doc_id,
        title=title,
        source_path=str(path),
        page_sizes=page_sizes,
        blocks=blocks,
        meta={
            "n_pages": len(page_sizes),
            "n_blocks": len(blocks),
            "n_tables": n_tables,
            "n_figures": n_figs,
            "body_size": body,
            "scanned_pages": scanned_pages,
            "report_year": report_year,
            "parse_seconds": round(time.time() - t0, 2),
        },
    )
    doc.close()
    return document


def parse_and_save(path: str | Path, settings: Settings = SETTINGS, **kw) -> Document:
    document = parse_pdf(path, settings, **kw)
    document.save(settings.parsed_dir / f"{document.doc_id}.json")
    return document


def load_document(doc_id: str, settings: Settings = SETTINGS) -> Document:
    return Document.load(settings.parsed_dir / f"{doc_id}.json")
