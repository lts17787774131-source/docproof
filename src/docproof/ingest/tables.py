"""表格抽取：从单元格几何还原 rowspan/colspan，并产出三种视图。

真实年报的表头几乎都是**合并单元格**（"本期金额/比例" 横跨两列，下面再分 "金额/比例"）。
早期版本直接按格子取文本，结果列名是空的（``列名： | 项目 |  |  |``），
下游的行式序列化于是拿不到列名、表格算子选错列。

这里做三件事：
1. 用单元格矩形的**相等性**判断合并：同一矩形出现在相邻列/行 → colspan/rowspan；
2. 把合并区域"填满"（每个被覆盖的格子都拿到锚点文本），保证每列都有名字；
3. 多行表头按 "/" 拼接成完整列名（``本期金额/比例/金额``）。
"""
from __future__ import annotations

import re

import pymupdf

from ..ir import Block, clean_cell

THOUSAND = re.compile(r"(?<=\d),(?=\d{3}\b)")


def _as_box(rect) -> tuple[float, float, float, float] | None:
    """兼容 Rect 与 (x0,y0,x1,y1) 元组两种单元格表示。"""
    if rect is None:
        return None
    if hasattr(rect, "x0"):
        return (float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1))
    try:
        x0, y0, x1, y1 = rect
        return (float(x0), float(y0), float(x1), float(y1))
    except Exception:
        return None


def _cell_text(page: pymupdf.Page, box: tuple[float, float, float, float]) -> str:
    txt = page.get_text("text", clip=box)
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\s*\n\s*", " ", txt)
    return clean_cell(txt.strip())


def _is_numberish(s: str) -> bool:
    return bool(re.fullmatch(r"[-+]?\d+(?:\.\d+)?%?", s.replace(",", "").strip()))


def _build_grid(page: pymupdf.Page, table, n_cols: int):
    """返回 (raw 网格, 每个锚点的 span 信息, 行数)。raw[r][c] = 单元格 key 或 None。"""
    rows = list(getattr(table, "rows", []) or [])
    n_rows = len(rows)
    raw: list[list[tuple | None]] = [[None] * n_cols for _ in range(n_rows)]
    text_of: dict[tuple, str] = {}

    for r, row in enumerate(rows):
        cells = list(getattr(row, "cells", []) or [])
        for c in range(min(n_cols, len(cells))):
            box = _as_box(cells[c])
            if box is None:
                continue
            key = tuple(round(v, 1) for v in box)
            if key not in text_of:
                text_of[key] = _cell_text(page, box)
            raw[r][c] = key

    span: dict[tuple, tuple[int, int]] = {}
    for r in range(n_rows):
        for c in range(n_cols):
            key = raw[r][c]
            if key is None or key in span:
                continue
            c2 = c
            while c2 + 1 < n_cols and raw[r][c2 + 1] == key:
                c2 += 1
            r2 = r
            while r2 + 1 < n_rows and raw[r2 + 1][c] == key:
                r2 += 1
            span[key] = (r2 - r + 1, c2 - c + 1)
    return raw, text_of, span, n_rows


def _fill(raw, text_of, span, n_rows, n_cols) -> list[list[str]]:
    """把合并区域填满：被覆盖的格子也拿到锚点文本。"""
    grid = [[""] * n_cols for _ in range(n_rows)]
    for r in range(n_rows):
        for c in range(n_cols):
            key = raw[r][c]
            if key is None:
                continue
            grid[r][c] = text_of.get(key, "")
    return grid


def _html(raw, text_of, span, n_rows, n_cols) -> str:
    covered = [[False] * n_cols for _ in range(n_rows)]
    out = ["<table>"]
    for r in range(n_rows):
        tds = []
        for c in range(n_cols):
            if covered[r][c]:
                continue
            key = raw[r][c]
            if key is None:
                covered[r][c] = True
                tds.append("<td></td>")
                continue
            rspan, cspan = span.get(key, (1, 1))
            for rr in range(r, min(n_rows, r + rspan)):
                for cc in range(c, min(n_cols, c + cspan)):
                    covered[rr][cc] = True
            attrs = (f' rowspan="{rspan}"' if rspan > 1 else "") + \
                    (f' colspan="{cspan}"' if cspan > 1 else "")
            tag = "th" if r == 0 else "td"
            val = (text_of.get(key, "") or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            tds.append(f"<{tag}{attrs}>{val}</{tag}>")
        out.append("<tr>" + "".join(tds) + "</tr>")
    out.append("</table>")
    return "".join(out)


def _header_size(grid: list[list[str]], max_header: int = 3) -> int:
    """开头连续多少行不含数字 → 认为是表头（真实年报常见 2 行表头）。"""
    n = 0
    for row in grid[:max_header]:
        if any(_is_numberish(cell) for cell in row if cell):
            break
        if any(cell.strip() for cell in row):
            n += 1
    return max(1, n)


def _split_header(grid: list[list[str]], n_header: int) -> tuple[list[str], list[list[str]]]:
    n_cols = len(grid[0]) if grid else 0
    header, data = [], []
    for c in range(n_cols):
        parts: list[str] = []
        for r in range(n_header):
            v = grid[r][c].strip()
            if v and (not parts or parts[-1] != v):
                parts.append(v)
        header.append("/".join(parts))
    for r in range(n_header, len(grid)):
        if any(cell.strip() for cell in grid[r]):
            data.append(grid[r])
    return header, data


def _header_by_overlap(raw, text_of, n_header: int, n_rows: int, n_cols: int) -> list[str]:
    """按**x 坐标重叠**把表头名映射到数据列。

    合并表头常与数据列错位一格（表头矩形横跨 7-9 列，而数值落在第 6 列），
    按列序号取名字会得到"附注=485886.24；2025 年=723120.77"这种自信的错误。
    """
    from statistics import median

    col_x: dict[int, tuple[float, float]] = {}
    for c in range(n_cols):
        xs = [(raw[r][c][0], raw[r][c][2]) for r in range(n_header, n_rows)
              if c < n_cols and raw[r][c] is not None]
        if xs:
            col_x[c] = (median(x[0] for x in xs), median(x[1] for x in xs))
    if not col_x:
        return [""] * n_cols

    head = [""] * n_cols
    for r in range(min(n_header, n_rows)):
        for c in range(n_cols):
            key = raw[r][c]
            if key is None:
                continue
            hx0, hx1, name = key[0], key[2], text_of.get(key, "").strip()
            if not name:
                continue
            for dc, (dx0, dx1) in col_x.items():
                overlap = min(hx1, dx1) - max(hx0, dx0)
                width = max(dx1 - dx0, 1e-6)
                if overlap > 0.5 * width and not head[dc]:
                    head[dc] = name
    return head


def extract_tables(page: pymupdf.Page, page_no: int, doc_id: str) -> list[Block]:
    width, height = page.rect.width, page.rect.height
    try:
        finder = page.find_tables()
    except Exception:
        return []

    out: list[Block] = []
    for t_idx, table in enumerate(getattr(finder, "tables", []) or []):
        n_cols = int(getattr(table, "col_count", 0) or 0)
        if n_cols < 2:
            continue
        raw, text_of, span, n_rows = _build_grid(page, table, n_cols)
        if n_rows < 2:
            continue
        grid = _fill(raw, text_of, span, n_rows, n_cols)
        grid = [r for r in grid if any(c.strip() for c in r)]
        if len(grid) < 2:
            continue
        n_header = _header_size(grid)
        header, data = _split_header(grid, n_header)
        if not data:
            continue
        header = _header_by_overlap(raw, text_of, n_header, n_rows, n_cols)

        x0, y0, x1, y1 = table.bbox
        cells = []
        for key, text in text_of.items():
            cells.append({"bbox": [key[0] / width, key[1] / height,
                                   key[2] / width, key[3] / height], "text": text})
        out.append(
            Block(
                id=f"{doc_id}:t{page_no}_{t_idx}",
                doc_id=doc_id,
                page=page_no,
                type="table",
                bbox=(x0 / width, y0 / height, x1 / width, y1 / height),
                text="\n".join(" | ".join(r) for r in grid),
                html=_html(raw, text_of, span, n_rows, n_cols),
                cells=cells,
                meta={
                    "n_rows": len(grid), "n_cols": n_cols, "merged_cells": len(span),
                    "header": header, "rows": data, "n_header_rows": n_header,
                },
            )
        )
    return out


def attach_table_captions(tables: list[Block], texts: list[Block]) -> None:
    """把表格上方最近的"表 N"文本块挂成 caption。"""
    caps = [b for b in texts if re.match(r"^\s*表", b.text)]
    for tb in tables:
        best, best_dy = None, 1e9
        for c in caps:
            if c.page != tb.page:
                continue
            dy = tb.bbox[1] - c.bbox[3]
            if -0.02 <= dy <= 0.06 and dy < best_dy:
                best, best_dy = c, dy
        if best is not None:
            tb.caption = best.text.strip().replace("\n", " ")
            best.type = "caption"
