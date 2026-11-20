"""向量化：通过 Ollama 调用 bge-m3（1024 维），带本地缓存。

不用 sentence-transformers 的原因：8G 卡上少一个 2.2G 的 torch 常驻进程，
而 Ollama 已经常驻在跑生成模型；向量化走同一个服务，显存账本最简单。
"""
from __future__ import annotations

import hashlib
import sqlite3
import threading
from pathlib import Path

import httpx
import numpy as np


class OllamaEmbedder:
    def __init__(self, url: str, model: str = "bge-m3",
                 cache_path: Path | None = None, batch_size: int = 16):
        self.url = url.rstrip("/")
        self.model = model
        self.batch_size = batch_size
        self.cache_path = cache_path
        self._conn: sqlite3.Connection | None = None
        # FastAPI 用线程池跑同步接口：连接是跨线程复用的，
        # 默认 check_same_thread=True 会在第二次上传时直接抛 ProgrammingError
        self._lock = threading.Lock()
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(cache_path), check_same_thread=False)
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS embed_cache("
                "key TEXT PRIMARY KEY, model TEXT, dim INTEGER, vec BLOB)")
            self._conn.commit()

    # ---- 缓存 ----
    def _key(self, text: str) -> str:
        return hashlib.sha1(f"{self.model}\x00{text}".encode("utf-8")).hexdigest()

    def _cache_get(self, key: str) -> np.ndarray | None:
        if self._conn is None:
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT vec, dim FROM embed_cache WHERE key=?", (key,)).fetchone()
        if not row:
            return None
        return np.frombuffer(row[0], dtype=np.float32).reshape(row[1])

    def _cache_put(self, key: str, vec: np.ndarray) -> None:
        if self._conn is None:
            return
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO embed_cache(key, model, dim, vec) VALUES(?,?,?,?)",
                (key, self.model, int(vec.shape[0]), vec.astype(np.float32).tobytes()))
            self._conn.commit()

    # ---- 调用 ----
    def _call(self, texts: list[str]) -> np.ndarray:
        with httpx.Client(base_url=self.url, timeout=180.0) as client:
            resp = client.post("/api/embed", json={"model": self.model, "input": texts})
            resp.raise_for_status()
            data = resp.json()
        vecs = np.asarray(data["embeddings"], dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        return vecs

    def embed(self, texts: list[str], use_cache: bool = True) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1024), dtype=np.float32)
        out: list[np.ndarray | None] = [None] * len(texts)
        todo: list[int] = []
        if use_cache and self._conn is not None:
            for i, t in enumerate(texts):
                v = self._cache_get(self._key(t))
                if v is None:
                    todo.append(i)
                else:
                    out[i] = v
        else:
            todo = list(range(len(texts)))

        for start in range(0, len(todo), self.batch_size):
            idxs = todo[start:start + self.batch_size]
            vecs = self._call([texts[i] for i in idxs])
            for j, i in enumerate(idxs):
                out[i] = vecs[j]
                if use_cache:
                    self._cache_put(self._key(texts[i]), vecs[j])

        mat = np.vstack([v if v is not None else np.zeros(1024, dtype=np.float32) for v in out])
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (mat / norms).astype(np.float32)


def cosine_scores(mat: np.ndarray, query: np.ndarray) -> np.ndarray:
    q = query.reshape(-1)
    qn = q / (np.linalg.norm(q) or 1.0)
    return mat @ qn
