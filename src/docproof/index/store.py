"""SQLite 存储：文档、块、向量一张库搞定（对应 ADR：pgvector 的轻量替代）。

为什么是 SQLite 而不是 Postgres+pgvector：
- 单机 MVP，零运维，一条命令起；
- 表结构（metadata 过滤 + 向量）完全可比，迁移到 pgvector 只需换 DAO；
- 面试时能讲清"我为什么先选 SQLite，以及什么规模该换"。
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

from ..ir import Chunk, Document

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents(
    doc_id TEXT PRIMARY KEY, title TEXT, source_path TEXT,
    n_pages INTEGER, meta TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS chunks(
    id TEXT PRIMARY KEY, doc_id TEXT, idx INTEGER, page INTEGER, ctype TEXT,
    section TEXT, text TEXT, html TEXT, blocks TEXT, parent_id TEXT,
    parent_text TEXT, bbox TEXT, meta TEXT
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE TABLE IF NOT EXISTS embeddings(
    chunk_id TEXT, model TEXT, dim INTEGER, vec BLOB,
    PRIMARY KEY(chunk_id, model)
);
"""


def connect(db_path: Path | str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def upsert_document(conn: sqlite3.Connection, doc: Document) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO documents(doc_id,title,source_path,n_pages,meta) VALUES(?,?,?,?,?)",
        (doc.doc_id, doc.title, doc.source_path, len(doc.page_sizes),
         json.dumps(doc.meta, ensure_ascii=False)))
    conn.commit()


def replace_chunks(conn: sqlite3.Connection, doc_id: str, chunks: list[Chunk]) -> None:
    conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
    conn.execute("DELETE FROM embeddings WHERE chunk_id LIKE ?", (f"{doc_id}:%",))
    conn.executemany(
        "INSERT INTO chunks(id,doc_id,idx,page,ctype,section,text,html,blocks,parent_id,parent_text,bbox,meta)"
        " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [(c.id, c.doc_id, c.idx, c.page, c.type, c.section, c.text, c.html,
          json.dumps(c.blocks, ensure_ascii=False), c.parent_id, c.parent_text,
          json.dumps(list(c.bbox)), json.dumps(c.meta, ensure_ascii=False)) for c in chunks])
    conn.commit()


def fetch_chunks(conn: sqlite3.Connection, doc_id: str) -> list[Chunk]:
    rows = conn.execute(
        "SELECT id,doc_id,idx,page,ctype,section,text,html,blocks,parent_id,parent_text,bbox,meta"
        " FROM chunks WHERE doc_id=? ORDER BY idx", (doc_id,)).fetchall()
    out = []
    for r in rows:
        out.append(Chunk(
            id=r[0], doc_id=r[1], idx=r[2], page=r[3], type=r[4], section=r[5],
            text=r[6], html=r[7], blocks=json.loads(r[8]), parent_id=r[9],
            parent_text=r[10], bbox=tuple(json.loads(r[11])), meta=json.loads(r[12])))
    return out


def list_documents(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT d.doc_id,d.title,d.n_pages,d.meta,(SELECT COUNT(*) FROM chunks c WHERE c.doc_id=d.doc_id)"
        " FROM documents d ORDER BY d.created_at DESC").fetchall()
    return [{"doc_id": r[0], "title": r[1], "n_pages": r[2],
             "meta": json.loads(r[3] or "{}"), "n_chunks": r[4]} for r in rows]


def get_document(conn: sqlite3.Connection, doc_id: str) -> dict | None:
    row = conn.execute(
        "SELECT doc_id,title,n_pages,meta,source_path FROM documents WHERE doc_id=?",
        (doc_id,)).fetchone()
    if not row:
        return None
    return {"doc_id": row[0], "title": row[1], "n_pages": row[2],
            "meta": json.loads(row[3] or "{}"), "source_path": row[4]}


def save_embeddings(conn: sqlite3.Connection, chunk_ids: list[str],
                    vecs: np.ndarray, model: str) -> None:
    conn.executemany(
        "INSERT OR REPLACE INTO embeddings(chunk_id,model,dim,vec) VALUES(?,?,?,?)",
        [(cid, model, int(v.shape[0]), v.astype(np.float32).tobytes())
         for cid, v in zip(chunk_ids, vecs)])
    conn.commit()


def load_embeddings(conn: sqlite3.Connection, doc_id: str,
                    model: str) -> tuple[list[str], np.ndarray | None]:
    rows = conn.execute(
        "SELECT e.chunk_id, e.vec, e.dim FROM embeddings e JOIN chunks c ON c.id=e.chunk_id"
        " WHERE c.doc_id=? AND e.model=? ORDER BY c.idx", (doc_id, model)).fetchall()
    if not rows:
        return [], None
    ids = [r[0] for r in rows]
    mat = np.vstack([np.frombuffer(r[1], dtype=np.float32).reshape(r[2]) for r in rows])
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return ids, (mat / norms).astype(np.float32)


def delete_index(conn: sqlite3.Connection, doc_id: str) -> int:
    """只删索引（块 + 向量），保留 documents 行——重建要靠它记着源文件路径。

    返回删掉的块数。文档会变成"未索引"状态，列表里仍可见、可重建。
    """
    ids = [r[0] for r in conn.execute("SELECT id FROM chunks WHERE doc_id=?", (doc_id,))]
    if ids:
        conn.executemany("DELETE FROM embeddings WHERE chunk_id=?", [(i,) for i in ids])
    conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
    conn.commit()
    return len(ids)


def purge_document(conn: sqlite3.Connection, doc_id: str) -> int:
    """彻底移除：索引 + 文档记录。源 PDF 不删（可能还要用）。"""
    n = delete_index(conn, doc_id)
    conn.execute("DELETE FROM documents WHERE doc_id=?", (doc_id,))
    conn.commit()
    return n
