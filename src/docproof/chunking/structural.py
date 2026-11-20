"""结构感知切分（主贡献之一）.

对照基线 B0 是"定长 512 字切分"，本模块要证明三件事：
1. 标题不单独成块（并入下文），否则标题会污染检索；
2. 表格整表一块、表头随行（超长表按行切但重复表头）；
3. 跨页续表合并（"（续上页）表 3"→ 并回上一张表）。

同时产出 parent_text 实现 small-to-big：子块（~420 字）用于召回，
父块（~1600 字）送给生成模型，兼顾"检索准"和"上下文全"。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..config import SETTINGS, Settings
from ..ir import Chunk, Document, html_to_text, table_rows_text, union_bbox

SENT_RE = re.compile(r"(?<=[。！？；!?;])")
CONT_RE = re.compile(r"^\s*[（(]\s*续\s*上?\s*页\s*[)）]\s*表\s*(\d+)")


@dataclass
class Unit:
    type: str
    text: str
    page: int
    section: str
    blocks: list[str] = field(default_factory=list)
    bbox: tuple = (0.0, 0.0, 1.0, 1.0)
    html: str = ""
    caption: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.text)


def _split_len(text: str, max_len: int) -> list[str]:
    """按句子边界切成 <= max_len 的片段；单句超长则硬切。"""
    parts = [p for p in SENT_RE.split(text) if p.strip()]
    out, buf = [], ""
    for p in parts:
        if len(buf) + len(p) <= max_len:
            buf += p
        else:
            if buf:
                out.append(buf.strip())
            while len(p) > max_len:
                out.append(p[:max_len])
                p = p[max_len:]
            buf = p
    if buf.strip():
        out.append(buf.strip())
    return out or [text[:max_len]]


def _table_text(block, table_view: str) -> str:
    if table_view == "rows":
        body = table_rows_text(block.html,
                               header=block.meta.get("header"),
                               rows=block.meta.get("rows"))
    elif table_view == "flat":
        body = block.text
    else:
        body = html_to_text(block.html)
    head = f"{block.caption}\n" if block.caption else ""
    return (head + body).strip()


def _units_structural(doc: Document, settings: Settings, table_view: str = "html") -> list[Unit]:
    units: list[Unit] = []
    buf: list = []
    heading_buf: list = []

    def _join(blocks: list) -> str:
        """拼接同一段落的多个块。

        两种版式要区别对待：
        - 正文段落：换行是排版换行，直接拼（否则"…星湖街/199 号"会被截断）；
        - 表单/简历：每一行是一个独立字段（"手 机：…" "邮 箱：…"），
          必须保留换行，否则所有字段被拼成一句，答案会糊成一团。
        """
        out = ""
        for b in blocks:
            lines = [ln.strip() for ln in b.text.split("\n") if ln.strip()]
            # 表单特征：多行、且多数行是"字段：值"或项目符号（简历/信息表）
            field_like = sum(("：" in ln or ":" in ln or ln.startswith(("•", "-", "·")))
                             for ln in lines)
            form_like = len(lines) >= 3 and field_like >= max(2, len(lines) * 0.5)
            if form_like:
                if out and not out.endswith("\n"):
                    out += "\n"
                out += "\n".join(lines)
            elif len(b.text) <= 40 and ("：" in b.text or b.text.startswith(("•", "-", "·"))):
                # 简历里每个字段常常是**独立文本块**（不是同块多行），
                # 也要换行，否则"姓名…手 机…邮 箱…"会糊成一句
                if out and not out.endswith("\n"):
                    out += "\n"
                out += b.text.strip()
            else:
                # 正文跨块必须无缝拼接，否则"…星湖街" / "199 号"会被切成两句
                out += b.text.replace("\n", "")
        return out

    def flush_headings() -> None:
        """连续多个标题 = 目录型内容（论文目录、年报目录页），要单独成块。

        单个标题不建索引（它已经进了 section_path），否则每节多一个噪声块；
        但目录页的每一行都会被判成标题，如果全丢掉，问"目录"就永远检索不到。
        """
        if len(heading_buf) < 2:
            heading_buf.clear()
            return
        text = "\n".join(h.text.strip() for h in heading_buf if h.text.strip())
        for piece in _split_len(text, settings.child_size):
            units.append(Unit(
                type="text", text=piece, page=heading_buf[0].page,
                section=heading_buf[0].section,
                blocks=[h.id for h in heading_buf],
                bbox=union_bbox([h.bbox for h in heading_buf]),
                extra={"toc": True},
            ))
        heading_buf.clear()

    def flush() -> None:
        if not buf:
            return
        text = _join(buf)
        pages = {b.page for b in buf}
        for piece in _split_len(text, settings.child_size):
            units.append(Unit(
                type="text", text=piece, page=min(pages),
                section=buf[0].section, blocks=[b.id for b in buf],
                bbox=union_bbox([b.bbox for b in buf]),
            ))
        buf.clear()

    for b in doc.blocks:
        if b.type in ("footer",):
            continue
        # 跨页就断开：否则一个块横跨两页，引用只能落到起始页，
        # 用户点"目录"却高亮了封面（实测问题）
        if (buf and b.page != buf[0].page) or (heading_buf and b.page != heading_buf[0].page):
            flush()
            flush_headings()
        if b.type == "heading":
            flush()
            heading_buf.append(b)
            continue
        flush_headings()
        if b.type == "table":
            flush()
            text = _table_text(b, table_view)
            units.append(Unit(type="table", text=text, page=b.page, section=b.section,
                              blocks=[b.id], bbox=b.bbox, html=b.html, caption=b.caption,
                              extra={k: b.meta[k] for k in ("subject", "statement") if k in b.meta}))
            continue
        if b.type == "figure":
            flush()
            # 没有图注、也没有附着文字的图块（背景图/装饰块）不建索引：
            # 它们只会变成"[图]"这样的噪声块，把检索结果挤掉
            text = (b.caption or b.text or "").strip()
            if not text:
                continue
            units.append(Unit(type="figure", text=text, page=b.page, section=b.section,
                              blocks=[b.id], bbox=b.bbox, caption=b.caption))
            continue
        buf.append(b)
        if sum(len(x.text) for x in buf) >= settings.child_size:
            flush()
    flush()
    flush_headings()
    return units


def merge_continuations(units: list[Unit]) -> list[Unit]:
    """把"（续上页）表 N"这一块并回上一张表，缓解 E9（跨页表被拆散）。"""
    drop: set[int] = set()
    for i, u in enumerate(units):
        m = CONT_RE.match(u.text)
        if not m:
            continue
        num = m.group(1)
        for j in range(i - 1, -1, -1):
            if units[j].type != "table":
                continue
            cap = units[j].caption or ""
            if re.search(rf"表\s*{num}\b", cap) or f"表 {num}" in units[j].text[:80]:
                units[j].text = units[j].text + "\n" + u.text
                units[j].blocks = units[j].blocks + u.blocks
                units[j].bbox = union_bbox([units[j].bbox, u.bbox])
                drop.add(i)
                # 续表后续的"行式"文本块（"2023 | 7.3 亿元"）也要一起并进来，
                # 否则只并了续页标题、没并数据行（实测就是这样丢掉 2023 那行的）
                k = i + 1
                while k < len(units) and units[k].type == "text" and "|" in units[k].text:
                    units[j].text = units[j].text + "\n" + units[k].text
                    units[j].blocks = units[j].blocks + units[k].blocks
                    units[j].bbox = union_bbox([units[j].bbox, units[k].bbox])
                    drop.add(k)
                    k += 1
                # 续页被识别成"独立的表"时（表头相同），把它的数据行并回来
                if k < len(units) and units[k].type == "table":
                    head = units[k].text.split("\n")[0].strip()
                    if head and head in units[j].text:
                        units[j].text = units[j].text + "\n" + units[k].text
                        units[j].blocks = units[j].blocks + units[k].blocks
                        units[j].bbox = union_bbox([units[j].bbox, units[k].bbox])
                        drop.add(k)
            break
    return [u for i, u in enumerate(units) if i not in drop]


def _units_fixed(doc: Document, settings: Settings) -> list[Unit]:
    """基线：把所有文本按定长切分，不认表格/标题（B0）。"""
    pieces: list[tuple[int, str, str]] = []  # (page, section, text)
    for b in doc.blocks:
        if b.type in ("heading", "footer"):
            continue
        txt = b.text if b.type != "table" else b.text
        if txt.strip():
            pieces.append((b.page, b.section, txt.strip()))

    units: list[Unit] = []
    buf, pages, section = "", [], ""
    for page, sec, txt in pieces:
        if not section:
            section = sec
        buf += txt.replace("\n", "")
        pages.append(page)
        while len(buf) >= settings.fixed_size:
            cut = buf[:settings.fixed_size]
            units.append(Unit(type="text", text=cut.strip(), page=min(pages),
                              section=section, bbox=(0.0, 0.0, 1.0, 1.0)))
            buf = buf[settings.fixed_size:]
            pages = [pages[-1]]
    if buf.strip():
        units.append(Unit(type="text", text=buf.strip(), page=min(pages or [1]),
                          section=section, bbox=(0.0, 0.0, 1.0, 1.0)))
    return units


def build_chunks(doc: Document, mode: str = "structural",
                 settings: Settings = SETTINGS, table_view: str = "rows") -> list[Chunk]:
    if mode == "fixed":
        units = _units_fixed(doc, settings)
    else:
        units = merge_continuations(_units_structural(doc, settings, table_view=table_view))

    # 父块：同 section 连续 unit 拼接
    parent_of: dict[int, str] = {}
    parent_text: dict[str, str] = {}
    p_idx = -1
    last_section = None
    for i, u in enumerate(units):
        if u.section != last_section:
            p_idx += 1
            last_section = u.section
            pid = f"{doc.doc_id}:p{p_idx}"
            parent_text[pid] = ""
        parent_of[i] = f"{doc.doc_id}:p{p_idx}"
        if len(parent_text[parent_of[i]]) < settings.parent_size * 1.5:
            parent_text[parent_of[i]] += ("\n" if parent_text[parent_of[i]] else "") + u.text

    chunks: list[Chunk] = []
    for i, u in enumerate(units):
        cid = f"{doc.doc_id}:c{i}"
        chunks.append(Chunk(
            id=cid, doc_id=doc.doc_id, idx=i, page=u.page, type=u.type,
            section=u.section, text=u.text, html=u.html, blocks=u.blocks,
            parent_id=parent_of[i], parent_text=parent_text[parent_of[i]][: settings.parent_size],
            bbox=u.bbox, meta={"n_blocks": len(u.blocks), "caption": u.caption, **u.extra},
        ))
    return chunks
