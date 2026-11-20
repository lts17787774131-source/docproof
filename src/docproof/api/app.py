"""FastAPI 服务：上传 → 异步化前的同步建索引 → 提问 → 带 bbox 的引用。

MVP 阶段先同步建索引（一份 7 页 PDF 约 2-3 秒），
但接口已经按"返回 doc_id + 状态"设计，后面换成队列不用改前端。
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.requests import Request

from ..chunking.structural import build_chunks
from ..config import SETTINGS
from ..generate.answer import Answerer, doc_year_of
from ..generate.providers import get_provider
from ..index import store
from ..index.dense import OllamaEmbedder
from ..ingest.pipeline import parse_pdf
from ..retrieve.hybrid import Retriever

app = FastAPI(title="DocProof", version="0.1.0")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

_embedder: OllamaEmbedder | None = None


def embedder() -> OllamaEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = OllamaEmbedder(SETTINGS.ollama_url, SETTINGS.embed_model,
                                   cache_path=SETTINGS.data_dir / "embed_cache.db")
    return _embedder


def conn():
    return store.connect(SETTINGS.db_path)


class AskRequest(BaseModel):
    doc_id: str
    question: str
    provider: str = "extractive"
    subject: str = "auto"  # auto | consolidated | parent


@app.get("/")
def index(request: Request):
    docs = store.list_documents(conn())
    # Starlette >= 0.29 的签名是 (request, name, context)，旧写法会被当成模板名
    return templates.TemplateResponse(request=request, name="index.html", context={"docs": docs})


@app.get("/api/docs")
def list_docs():
    return store.list_documents(conn())


def _index_path(raw: Path) -> dict:
    """解析 → 切分 → 入库 → 向量。上传和"重建"共用这一段。"""
    t0 = time.time()
    doc = parse_pdf(raw, SETTINGS)
    chunks = build_chunks(doc, "structural", SETTINGS, table_view="rows")
    c = conn()
    store.upsert_document(c, doc)
    store.replace_chunks(c, doc.doc_id, chunks)
    vecs = embedder().embed([ch.text for ch in chunks])
    store.save_embeddings(c, [ch.id for ch in chunks], vecs, SETTINGS.embed_model)
    return {"doc_id": doc.doc_id, "title": doc.title, "pages": doc.meta["n_pages"],
            "chunks": len(chunks), "tables": doc.meta["n_tables"],
            "figures": doc.meta["n_figures"], "seconds": round(time.time() - t0, 2)}


@app.post("/api/docs")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "仅支持 PDF")
    SETTINGS.ensure_dirs()
    raw = SETTINGS.raw_dir / file.filename
    raw.write_bytes(await file.read())
    return _index_path(raw)


@app.delete("/api/docs/{doc_id}")
def delete_doc(doc_id: str, purge: bool = False):
    """默认只删索引（保留文档记录，可重建）；purge=true 连记录和页面图一起删。"""
    c = conn()
    if not store.get_document(c, doc_id):
        raise HTTPException(404, "文档不存在")
    n = store.purge_document(c, doc_id) if purge else store.delete_index(c, doc_id)
    if purge:
        pages = SETTINGS.pages_dir / doc_id
        if pages.exists():
            shutil.rmtree(pages, ignore_errors=True)
    return {"doc_id": doc_id, "deleted_chunks": n, "purged": purge}


@app.post("/api/docs/{doc_id}/rebuild")
def rebuild_doc(doc_id: str):
    c = conn()
    info = store.get_document(c, doc_id)
    if not info:
        raise HTTPException(404, "文档不存在")
    src = Path(info.get("source_path") or "")
    if not src.exists():
        raise HTTPException(404, f"源文件不存在：{src.name}（删索引时不会删 PDF，但文件可能被移走了）")
    return _index_path(src)


@app.post("/api/ask")
def ask(req: AskRequest):
    c = conn()
    chunks = store.fetch_chunks(c, req.doc_id)
    if not chunks:
        raise HTTPException(404, "文档未索引")
    _, mat = store.load_embeddings(c, req.doc_id, SETTINGS.embed_model)
    retriever = Retriever(chunks, embedder=embedder(),
                          embeddings=([ch.id for ch in chunks], mat) if mat is not None else None)
    answerer = Answerer(retriever, get_provider(req.provider, SETTINGS), SETTINGS)
    meta = store.get_document(c, req.doc_id) or {}
    payload = answerer.ask(req.question, doc_id=req.doc_id,
                           doc_year=doc_year_of(meta.get("meta"), meta.get("title", ""), chunks),
                           subject=None if req.subject in ("", "auto") else req.subject)
    return payload.to_dict()


@app.get("/api/docs/{doc_id}/pages/{page}.png")
def page_image(doc_id: str, page: int):
    p = SETTINGS.pages_dir / doc_id / f"p{page}.png"
    if not p.exists():
        raise HTTPException(404, "page not found")
    return FileResponse(p)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=7860, log_level="info")
