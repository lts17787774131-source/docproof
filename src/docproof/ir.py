"""DocIR：文档中间表示.

设计原则（对应 ADR-D1/D2）：只保留三类用途需要的字段——
检索（text / section / type）、引用（page / bbox / block_id）、展示（html / cells）。
所有下游模块（切分、索引、检索、生成、前端）只认这一层，不直接碰 PDF。
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

BLOCK_TYPES = ("heading", "text", "table", "figure", "caption", "footer")

BBox = tuple[float, float, float, float]  # 归一化坐标 (x0, y0, x1, y1)，取值 0-1


@dataclass
class Block:
    id: str
    doc_id: str
    page: int
    type: str
    bbox: BBox
    text: str = ""
    html: str = ""
    cells: list[dict] = field(default_factory=list)
    caption: str = ""
    section: str = ""
    order: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["bbox"] = list(self.bbox)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Block":
        d = dict(d)
        d["bbox"] = tuple(d["bbox"])
        return cls(**d)


@dataclass
class Document:
    doc_id: str
    title: str
    source_path: str
    page_sizes: list[list[float]]  # [[w, h], ...] 单位 pt
    blocks: list[Block] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    # ---- 查询 ----
    def page_size(self, page: int) -> tuple[float, float]:
        if 1 <= page <= len(self.page_sizes):
            w, h = self.page_sizes[page - 1]
            return float(w), float(h)
        return 595.0, 842.0

    def block(self, block_id: str) -> Block | None:
        for b in self.blocks:
            if b.id == block_id:
                return b
        return None

    def sections(self) -> list[str]:
        seen, out = set(), []
        for b in self.blocks:
            if b.section and b.section not in seen:
                seen.add(b.section)
                out.append(b.section)
        return out

    # ---- 持久化 ----
    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "source_path": self.source_path,
            "page_sizes": self.page_sizes,
            "blocks": [b.to_dict() for b in self.blocks],
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Document":
        return cls(
            doc_id=d["doc_id"],
            title=d["title"],
            source_path=d["source_path"],
            page_sizes=d["page_sizes"],
            blocks=[Block.from_dict(b) for b in d["blocks"]],
            meta=d.get("meta", {}),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Document":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


@dataclass
class Chunk:
    """检索单元：子块用于召回，parent_text 用于生成（small-to-big）。"""

    id: str
    doc_id: str
    idx: int
    page: int
    type: str
    section: str
    text: str
    html: str = ""
    blocks: list[str] = field(default_factory=list)
    parent_id: str = ""
    parent_text: str = ""
    bbox: BBox = (0.0, 0.0, 1.0, 1.0)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["bbox"] = list(self.bbox)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        d = dict(d)
        d["bbox"] = tuple(d["bbox"])
        return cls(**d)


def union_bbox(boxes: Iterable[BBox]) -> BBox:
    boxes = list(boxes)
    if not boxes:
        return (0.0, 0.0, 1.0, 1.0)
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)
    return (x0, y0, x1, y1)


def html_to_text(html: str) -> str:
    """把表格 HTML 摊平成"表头：值"的文本视图（供 BM25 使用）。

    与"整表存 HTML"的分工：HTML 给生成模型看结构，这段文本给检索看词。
    """
    if not html:
        return ""
    import re

    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, flags=re.S)
    grid: list[list[str]] = []
    for row in rows:
        cells = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row, flags=re.S)
        grid.append([re.sub(r"<[^>]+>", "", c).strip() for c in cells])
    if not grid:
        return ""
    header = grid[0]
    lines = [" | ".join(header)]
    for r in grid[1:]:
        lines.append(" | ".join(r))
    return "\n".join(lines)


def clean_cell(text: str) -> str:
    """单元格清洗：全角转半角、去千分位、括号负数转负号。

    真实年报里 ``（1,234.56）`` 表示负数、``586,499,360.82`` 是千分位；
    不归一化的话，表格算子把 ``1,234`` 当成两个数，算术结果必错。
    """
    import re
    import unicodedata

    s = unicodedata.normalize("NFKC", text or "").strip()
    s = re.sub(r"\s+", " ", s)
    m = re.fullmatch(r"\((\d[\d,]*(?:\.\d+)?)\)", s)
    if m:
        return "-" + m.group(1).replace(",", "")
    s = re.sub(r"\d[\d,]*(?:\.\d+)?", lambda mm: mm.group(0).replace(",", ""), s)
    return s


NOTE_RE = re.compile(r"^[一二三四五六七八九十]+、?\s*[（(][^）)]{1,8}[)）]$"
                     r"|^附注|^注\s*[：:]?\s*\d*$")


NUM_WITH_UNIT = re.compile(r"^[-+]?[\d,]+(?:\.\d+)?\s*(?:亿元|万元|元|%|％|股|人|项|个)?$")


def is_numberish(v: str) -> bool:
    """数值单元格：允许单位后缀（"5.6 亿元"），否则"年份|金额"两列的表会把标签判错。"""
    v = (v or "").strip()
    if v in ("-", "—", "－", "/", ""):
        return v != ""
    return bool(NUM_WITH_UNIT.match(v))


def propagate_header(header: list[str]) -> list[str]:
    """表头扩散：合并单元格常常只在一列留下名字，其余是空。

    对每个空列，取**左右最近的非空表头**（等距优先左），
    这样 "['', '项目', '', '', '附注', '', '', '2025 年', ...]" 能补成
    "['项目','项目','项目','附注','附注','附注','2025 年',...]"。
    """
    n = len(header)
    filled = [""] * n
    for i, h in enumerate(header):
        if h and h.strip():
            filled[i] = h.strip()
    for i in range(n):
        if filled[i]:
            continue
        left = right = None
        for d in range(1, n):
            if i - d >= 0 and filled[i - d]:
                left = (d, filled[i - d])
                break
        for d in range(1, n):
            if i + d < n and filled[i + d]:
                right = (d, filled[i + d])
                break
        if left and right:
            filled[i] = left[1] if left[0] <= right[0] else right[1]
        elif left:
            filled[i] = left[1]
        elif right:
            filled[i] = right[1]
    if len({h for h in filled if h}) < 2:
        filled = [f"列{i + 1}" for i in range(n)]
    return filled


def rows_text(header: list[str], rows: list[list[str]]) -> str:
    """用结构化表头 + 数据行生成行式文本.

    真实财报主表的两个坑（都与"列"有关）：
    1. **标签列会左右跳**：缩进子项在 col0，主项在 col1，不能写死第 0 列；
    2. **附注列**（"五、(三十八)"）是页码引用，混进数值列会污染答案。
    这里逐行挑"第一个非数值单元格"当标签，跳过附注列，其余按表头列名配对。
    """
    if not header and not rows:
        return ""
    n = max([len(header)] + [len(r) for r in rows]) if rows else len(header)
    head = propagate_header(list(header) + [""] * (n - len(header)))

    # 数值列：只要任意一行在该列是数字，就认为它是"值列"。
    # 按列位置映射，而不是按"第几个数字"映射——这样某年为空时不会错位。
    numeric_cols = [c for c in range(n) if any(is_numberish(r[c]) for r in rows if c < len(r))]
    names: list[str] = []
    for c in numeric_cols:
        nm = head[c] or f"列{c + 1}"
        if not names or names[-1] != nm:
            names.append(nm)
    if len(names) != len(numeric_cols):      # 表头不可靠，退化为列号
        names = [f"列{c + 1}" for c in numeric_cols]
    if not numeric_cols:
        return ""

    lines = ["列名：" + " | ".join(names)]
    for r in rows:
        row = list(r) + [""] * (n - len(r))
        cells = [(i, row[i].strip()) for i in range(n) if row[i].strip()]
        if not cells:
            continue
        label_i = cells[0][0]
        # 首列若是 1-3 位纯数字，通常是行号（财务报表常见），标签要让给下一列
        if re.fullmatch(r"\d{1,3}", row[label_i].strip()) and len(cells) > 1:
            label_i = cells[1][0]
        label = row[label_i].strip()
        pairs = []
        for i, v in cells:
            if i == label_i or i not in numeric_cols:
                continue
            k = numeric_cols.index(i)
            pairs.append(f"{names[k]}={v}")
        if not pairs:          # 纯分节标题行（"一、流动资产"）
            continue
        lines.append(f"{label}：" + "；".join(pairs))
    return "\n".join(lines)


def table_rows_text(html: str, header: list[str] | None = None,
                    rows: list[list[str]] | None = None) -> str:
    """表头随行的行式序列化：每行都自带列名，单独成句也能被检索/引用。

    这是 B2 的核心改动。扁平化（``html_to_text``）时，"第三季度" 和 "12.4"
    分散在不同的行里，模型和 BM25 都要靠猜；行式化之后，
    "第三季度：营业收入（亿元）=12.4" 本身就是一个自洽的事实句。

    传了 ``header`` / ``rows`` 就直接用（解析阶段已经还原了合并单元格），
    否则退回解析 HTML（兼容旧数据）。
    """
    if header is not None or rows is not None:
        return rows_text(header or [], rows or [])
    if not html:
        return ""
    import re

    trs = re.findall(r"<tr[^>]*>(.*?)</tr>", html, flags=re.S)
    grid: list[list[str]] = []
    for row in trs:
        cells = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row, flags=re.S)
        grid.append([re.sub(r"<[^>]+>", "", c).strip() for c in cells])
    if len(grid) < 2:
        return html_to_text(html)
    return rows_text(grid[0], grid[1:])
