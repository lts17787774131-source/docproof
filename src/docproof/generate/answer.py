"""问答编排：检索 → 证据覆盖检查 → 生成 → 引用校验 → 返回/拒答.

拒答不是"分数低就拒"（RRF 分数没有绝对意义），而是两道可解释的闸门：
1. **关键词覆盖**：问题里的实词有多少出现在 top 证据里；
2. **数字闸门**：问题里带年份/金额，而 top 证据里完全没有这个数字 → 基本可判"文档里没有"。
这两条在财报场景特别有效（"2025 年营收"这种问题会直接命中第二条）。
"""
from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field

from ..chunking.structural import Chunk
from ..config import SETTINGS, Settings
from ..index.bm25 import tokenize
from ..retrieve.hybrid import Hit, Retriever, route
from .providers import Evidence, Provider

NUM_RE = re.compile(r"\d+(?:\.\d+)?")


@dataclass
class Citation:
    eid: str
    chunk_id: str
    page: int
    bbox: tuple
    section: str
    snippet: str
    verified: bool = False
    type: str = "text"
    statement: str = ""
    subject: str = ""
    score: float = 0.0
    full_text: str = ""


@dataclass
class AnswerPayload:
    question: str
    answer: str
    refused: bool
    citations: list[Citation] = field(default_factory=list)
    provider: str = ""
    route: str = ""
    latency_ms: int = 0
    hits: list[dict] = field(default_factory=list)
    alternatives: list[dict] = field(default_factory=list)
    refusal_reason: str = ""
    doc_id: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["citations"] = [asdict(c) for c in self.citations]
        return d


def _coverage(question: str, chunks: list[Chunk]) -> float:
    q = [t for t in tokenize(question) if len(t) > 1 or t.isdigit()]
    if not q:
        return 1.0
    qs = set(q)
    best = 0.0
    for c in chunks:
        cs = set(tokenize(c.text))
        best = max(best, len(qs & cs) / len(qs))
    return best


def _numbers_missing(question: str, chunks: list[Chunk]) -> bool:
    nums = set(NUM_RE.findall(question))
    if not nums:
        return False
    blob = "".join(c.text for c in chunks)
    return not any(n in blob for n in nums)


def doc_year_of(meta: dict | None, title: str = "", chunks: list | None = None) -> int | None:
    """报告期年份：优先解析元数据，其次标题，最后用全文年份众数推断。

    老索引没有 report_year，而标题常常只有公司名（"广咨国际"），
    所以加了第三条兜底：整篇文档里出现最多的年份就是报告期。
    """
    if meta and meta.get("report_year"):
        return int(meta["report_year"])
    m = re.search(r"(20\d{2})\s*年", title or "")
    if m:
        return int(m.group(1))
    if chunks:
        from collections import Counter
        cnt: Counter[int] = Counter()
        for c in chunks[:400]:
            for y in re.findall(r"(20\d{2})\s*年", c.text):
                cnt[int(y)] += 1
        if cnt:
            return cnt.most_common(1)[0][0]
    return None


class Answerer:
    def __init__(self, retriever: Retriever, provider: Provider,
                 settings: Settings = SETTINGS):
        self.retriever = retriever
        self.provider = provider
        self.settings = settings

    def _hit_summary(self, hits: list[Hit]) -> list[dict]:
        return [{
            "chunk_id": h.chunk.id, "page": h.chunk.page, "type": h.chunk.type,
            "section": h.chunk.section, "score": round(h.score, 4),
            "text": h.chunk.text[:220],
        } for h in hits]

    def _alternatives(self, question: str, hits: list[Hit], limit: int = 5) -> list[dict]:
        """每个召回块给出"这一段里最像答案的一句"。

        用户诉求：问"名字"时把所有相关答案都列出来、逐条可定位。
        所以这里不只看模型选中的那一条，而是把 top-k 各自的最佳句都摊开。
        """
        picker = getattr(self.provider, "best_sentence", None)
        out: list[dict] = []
        seen: set[str] = set()
        for h in hits[:limit]:
            sent = picker(question, h.chunk.text) if callable(picker) else h.chunk.text[:200]
            sent = (sent or "").strip()
            if not sent or sent in seen:
                continue
            # 过滤掉"基本信息""项目经历"这类纯小标题，它们不是答案
            if len(sent) < 8 and "：" not in sent and not re.search(r"\d", sent):
                continue
            seen.add(sent)
            out.append({"text": sent, "page": h.chunk.page, "bbox": list(h.chunk.bbox),
                        "score": round(h.score, 4), "chunk_id": h.chunk.id,
                        "type": h.chunk.type, "section": h.chunk.section,
                        "block_text": h.chunk.text})
        return out

    def _refuse(self, question: str, hits: list[Hit], reason: str,
                t0: float, doc_id: str) -> AnswerPayload:
        return AnswerPayload(
            question=question,
            answer="根据当前文档无法确定该问题的答案。",
            refused=True, citations=[], provider=self.provider.name,
            route=route(question), latency_ms=int((time.time() - t0) * 1000),
            hits=self._hit_summary(hits), alternatives=self._alternatives(question, hits),
            refusal_reason=reason, doc_id=doc_id,
        )

    def ask(self, question: str, doc_id: str = "", top_k: int | None = None,
            use_dense: bool = True, type_boost: bool = True,
            doc_year: int | None = None, subject: str | None = None) -> AnswerPayload:
        t0 = time.time()
        # 文档是 2025 年报，问 2026/2027 的数据：直接拒，别让"三年行动计划"里的
        # 未来年份把数字闸门骗过去（真实测试里确实骗过了）
        years = [int(y) for y in re.findall(r"(20\d{2})\s*年", question)]
        if doc_year and years and max(years) > doc_year:
            return self._refuse(question, [], "future_year", t0, doc_id)

        top_k = top_k or self.settings.top_k
        hits = self.retriever.search(question, top_k=top_k, use_dense=use_dense,
                                     rrf_k=self.settings.rrf_k, type_boost=type_boost)
        if not hits:
            return self._refuse(question, [], "no_hits", t0, doc_id)

        probe = [h.chunk for h in hits[:3]]
        # 数字闸门放在最前面：问题里的年份/金额在证据里完全找不到，基本就是"文档没有"
        if _numbers_missing(question, probe):
            return self._refuse(question, hits, "number_absent", t0, doc_id)

        evidences = [Evidence(eid=f"E{i + 1}", chunk=h.chunk) for i, h in enumerate(hits)]
        result = self.provider.answer(question, evidences, subject=subject)
        if result.refused or not result.cited:
            cov = _coverage(question, probe)
            reason = "model_refused" if result.refused else f"low_coverage({cov:.2f})"
            return self._refuse(question, hits, reason, t0, doc_id)

        by_eid = {ev.eid: ev for ev in evidences}
        score_of = {h.chunk.id: round(h.score, 4) for h in hits}
        citations: list[Citation] = []
        for eid in result.cited:
            ev = by_eid.get(eid)
            if not ev:
                continue
            if result.derived and result.operands:
                # 计算型答案（"增长 3.2 亿元"）的 3.2 本来就不在原文里，
                # 该校验的是**依据的两个数**都在证据中，而不是结果数字
                verified = all(f"{x:g}" in ev.chunk.text for x in result.operands)
            else:
                answered_nums = set(NUM_RE.findall(result.text))
                verified = (not answered_nums) or all(n in ev.chunk.text for n in answered_nums)
            citations.append(Citation(
                eid=eid, chunk_id=ev.chunk.id, page=ev.chunk.page, bbox=ev.chunk.bbox,
                section=ev.chunk.section, snippet=ev.chunk.text[:300],
                verified=verified, type=ev.chunk.type,
                statement=ev.chunk.meta.get("statement", ""),
                subject=ev.chunk.meta.get("subject", ""),
                score=score_of.get(ev.chunk.id, 0.0),
                full_text=ev.chunk.text))

        return AnswerPayload(
            question=question, answer=result.text, refused=result.refused,
            citations=citations, provider=result.provider, route=route(question),
            latency_ms=int((time.time() - t0) * 1000),
            hits=self._hit_summary(hits), alternatives=self._alternatives(question, hits),
            doc_id=doc_id,
            refusal_reason="model_refused" if result.refused else "",
        )
