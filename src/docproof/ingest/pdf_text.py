"""版面与文本抽取：把一页 PDF 拆成可分类的 block（标题/正文/页眉页脚）。

不用深度学习版面模型（DocLayout-YOLO）而用字体/位置启发式，原因是：
1) 本项目 80% 的文档有文本层，启发式足够；
2) 少一个 2G 的模型依赖，8G 卡上更稳；
3) 规则可解释——面试被问"为什么这块被判成标题"能答上来。

关键坑：**字号基准必须是文档级的**。表内常用 8-9pt，若按页统计，
表多的页面会把"正文主字号"拉到 8.5pt，10.5pt 的正文就被误判成标题
（实测确实发生过）。
"""
from __future__ import annotations

import re
from collections import Counter

import pymupdf

from ..ir import Block

HEADING_RE = re.compile(
    r"^\s*(第[一二三四五六七八九十百]+[节章部]|[一二三四五六七八九十]+、|\d+(?:\.\d+){0,2}[、.\s])"
)
CAPTION_FIG_RE = re.compile(r"^\s*(图|figure|Figure)\s*\d+")
CAPTION_TAB_RE = re.compile(r"^\s*(表|table|Table)\s*\d+")
CONT_RE = re.compile(r"^\s*[（(]\s*续\s*上?\s*页\s*[)）]\s*表\s*(\d+)")


def _collect_sizes(page: pymupdf.Page, exclude: list[tuple], counter: Counter) -> None:
    for block in page.get_text("dict").get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = (span.get("text") or "").strip()
                if not text:
                    continue
                if exclude:
                    sx0, sy0, sx1, sy1 = span["bbox"]
                    cx, cy = (sx0 + sx1) / 2, (sy0 + sy1) / 2
                    if any(r[0] <= cx <= r[2] and r[1] <= cy <= r[3] for r in exclude):
                        continue
                counter[round(float(span["size"]), 1)] += len(text)


def body_font_size(page: pymupdf.Page, exclude: list[tuple] | None = None) -> float:
    counter: Counter[float] = Counter()
    _collect_sizes(page, exclude or [], counter)
    if not counter:
        return 10.0
    return float(counter.most_common(1)[0][0])


def global_body_size(pages: list[pymupdf.Page],
                     rects_per_page: list[list[tuple]] | None = None) -> float:
    """全文档按字符数加权的主字号。标题判断统一用它，避免逐页漂移。"""
    counter: Counter[float] = Counter()
    rects_per_page = rects_per_page or [[] for _ in pages]
    for page, rects in zip(pages, rects_per_page):
        _collect_sizes(page, rects, counter)
    if not counter:
        return 10.0
    return float(counter.most_common(1)[0][0])


def _block_text(block: dict) -> tuple[str, float, bool]:
    lines, sizes, bold = [], [], False
    for line in block.get("lines", []):
        parts = []
        for span in line.get("spans", []):
            t = span.get("text") or ""
            if t:
                parts.append(t)
                sizes.append(float(span["size"]))
                font = (span.get("font") or "").lower()
                if "bold" in font or "hei" in font or "黑体" in font:
                    bold = True
        if parts:
            lines.append("".join(parts).strip())
    text = "\n".join([ln for ln in lines if ln])
    return text.strip(), (max(sizes) if sizes else 10.0), bold


def heading_level(text: str) -> int:
    t = text.strip()
    if re.match(r"^第[一二三四五六七八九十百]+[节章部]", t):
        return 1
    if re.match(r"^[一二三四五六七八九十]+、", t):
        return 2
    m = re.match(r"^(\d+(?:\.\d+){0,2})[、.\s]", t)
    if m:
        return m.group(1).count(".") + 2
    return 2


def extract_text_blocks(page: pymupdf.Page, page_no: int, doc_id: str,
                        body_size: float | None = None,
                        exclude: list[tuple] | None = None) -> list[Block]:
    """返回标题 + 正文 + 页眉页脚，不含表格与图片。"""
    width, height = page.rect.width, page.rect.height
    body = body_size if body_size is not None else global_body_size([page], [exclude or []])
    out: list[Block] = []
    data = page.get_text("dict", sort=True)

    for b_idx, block in enumerate(data.get("blocks", [])):
        if block.get("type") != 0:
            continue
        text, max_size, bold = _block_text(block)
        if not text:
            continue
        x0, y0, x1, y1 = block["bbox"]
        norm = (x0 / width, y0 / height, x1 / width, y1 / height)

        is_footer = (y1 < height * 0.055 or y0 > height * 0.945) and len(text) < 60
        # 标题不含句读：否则"199 号，报告期末在职员工 5,806 人"这种以数字开头的
        # 正文会被当成标题，段落被拦腰截断（实测把地址和员工数都切没了）
        looks_like_prose = ("。" in text) or ("，" in text) or ("；" in text)
        is_heading = (not looks_like_prose) and (
            (HEADING_RE.match(text) and len(text) <= 40 and max_size >= body * 0.85)
            or (max_size >= body * 1.35 and len(text) <= 40 and text.count("\n") <= 1)
        )

        if is_footer:
            btype = "footer"
        elif is_heading:
            btype = "heading"
        elif CAPTION_FIG_RE.match(text) or CAPTION_TAB_RE.match(text) or CONT_RE.match(text):
            btype = "caption"
        else:
            btype = "text"

        out.append(
            Block(
                id=f"{doc_id}:b{page_no}_{b_idx}",
                doc_id=doc_id,
                page=page_no,
                type=btype,
                bbox=norm,
                text=text,
                meta={"font_size": round(max_size, 1), "bold": bold,
                      "body_size": body, "level": heading_level(text) if btype == "heading" else 0},
            )
        )
    return out


def page_char_count(page: pymupdf.Page) -> int:
    return len((page.get_text("text") or "").strip())
