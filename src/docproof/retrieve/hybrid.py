"""混合检索：BM25 + 向量，RRF 融合，外加一个"题型路由"的轻量先验。

RRF（Reciprocal Rank Fusion）不需要调权重，比加权求和稳：
    score(d) = Σ_r 1 / (k + rank_r(d))
路由不是炫技：财报题里"第三季度/毛利率/占比"基本都在表里，
"图 2/趋势"基本都在图里，给对应类型一个很小的加权即可。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from ..chunking.structural import Chunk
from ..index.bm25 import BM25
from ..index.dense import OllamaEmbedder, cosine_scores

TABLE_HINT = re.compile(r"表\s*\d+|季度|同比|环比|毛利率|占比|增长率|营业收入|净利润|现金流|资产")
FIGURE_HINT = re.compile(r"图\s*\d+|如图所示|趋势|示意图|变化情况")

# 领域同义词：真实用户很少说"会计师事务所"，多说"审计机构/审计所"
SYNONYMS = {
    "审计机构": ["会计师事务所", "审计机构", "审计所"],
    "审计所": ["会计师事务所", "审计机构"],
    "会计师": ["会计师事务所"],
    "营收": ["营业收入", "营收"],
    "净利": ["净利润", "归母净利润"],
    "员工": ["员工", "在职员工", "职工"],
    "研发": ["研发", "研发投入", "研发支出"],
    # 简历/表单类
    "手机号": ["手机", "电话", "联系方式"],
    "电话": ["手机", "联系方式", "电话"],
    "联系方式": ["手机", "电话", "邮箱"],
    "邮箱": ["邮箱", "邮件"],
    "学历": ["教育经历", "学历", "本科", "硕士"],
    "学校": ["教育经历", "大学", "学院"],
    "做过的项目": ["项目经历", "项目经验"],
    # 论文/英文文档：问"目录"要能命中 Contents 页，而不是返回摘要页
    "目录": ["目录", "目次", "contents", "table of contents"],
    "目次": ["目录", "contents"],
    "摘要": ["摘要", "abstract"],
    "参考文献": ["参考文献", "references", "bibliography"],
    "结论": ["结论", "conclusion", "conclusions"],
}


def expand(question: str) -> str:
    extra = []
    for key, alts in SYNONYMS.items():
        if key in question:
            extra.extend(a for a in alts if a not in question)
    return question + (" " + " ".join(dict.fromkeys(extra)) if extra else "")


def canonical(question: str) -> str:
    """把口语词替换成文档里的标准说法（用于打分，不适合做"追加"）。

    检索可以靠扩展词提高召回；但打分时如果只是"追加"同义词，
    分母变大、相似度被稀释，反而导致"检索到了却判不相关"。
    """
    q = question
    for key, alts in SYNONYMS.items():
        if key in q:
            q = q.replace(key, alts[0])
    return q


@dataclass
class Hit:
    chunk: Chunk
    score: float
    bm25_rank: int | None = None
    dense_rank: int | None = None


def route(question: str) -> str:
    if FIGURE_HINT.search(question):
        return "figure"
    if TABLE_HINT.search(question):
        return "table"
    return "text"


class Retriever:
    def __init__(self, chunks: list[Chunk], embedder: OllamaEmbedder | None = None,
                 embeddings: tuple[list[str], np.ndarray | None] | None = None):
        self.chunks = chunks
        self.by_id = {c.id: c for c in chunks}
        self.bm25 = BM25([c.text for c in chunks]) if chunks else None
        self.embedder = embedder
        self.dense_ids: list[str] = []
        self.dense_mat: np.ndarray | None = None
        if embeddings:
            self.dense_ids, self.dense_mat = embeddings

    # ---- 单路 ----
    def _bm25(self, question: str, n: int) -> list[tuple[str, float]]:
        if not self.bm25:
            return []
        return [(self.chunks[i].id, s) for i, s in self.bm25.search(question, n)]

    def _dense(self, question: str, n: int) -> list[tuple[str, float]]:
        if self.embedder is None or self.dense_mat is None or self.dense_mat.size == 0:
            return []
        qv = self.embedder.embed([question])[0]
        scores = cosine_scores(self.dense_mat, qv)
        order = np.argsort(-scores)[:n]
        return [(self.dense_ids[i], float(scores[i])) for i in order]

    # ---- 融合 ----
    def search(self, question: str, top_k: int = 8, use_bm25: bool = True,
               use_dense: bool = True, rrf_k: int = 60, type_boost: bool = True,
               bm25_weight: float = 1.0, dense_weight: float = 0.6) -> list[Hit]:
        if not self.chunks:
            return []
        depth = max(50, top_k * 4)
        ranks: dict[str, dict] = {}
        query = expand(question)

        if use_bm25:
            for r, (cid, s) in enumerate(self._bm25(query, depth)):
                ranks.setdefault(cid, {"bm25": r, "dense": None, "bm25s": s, "denses": 0.0})
                ranks[cid]["bm25s"] = s
        if use_dense:
            for r, (cid, s) in enumerate(self._dense(query, depth)):
                d = ranks.setdefault(cid, {"bm25": None, "dense": None, "bm25s": 0.0, "denses": 0.0})
                d["dense"] = r
                d["denses"] = s

        preferred = route(question) if type_boost else "text"
        hits: list[Hit] = []
        for cid, r in ranks.items():
            score = 0.0
            if r["bm25"] is not None:
                score += bm25_weight / (rrf_k + r["bm25"] + 1)
            if r["dense"] is not None:
                score += dense_weight / (rrf_k + r["dense"] + 1)
            ch = self.by_id[cid]
            if type_boost and ch.type == preferred:
                score *= 1.08
            hits.append(Hit(chunk=ch, score=score, bm25_rank=r["bm25"], dense_rank=r["dense"]))

        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]
