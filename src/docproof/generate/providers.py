"""答案生成：三种 provider + 引用校验 + 阈值拒答。

- ``extractive``：确定性抽取式（默认用于**消融实验**）。
  它把"检索到的证据里能不能找到答案"这件事变得可测量——如果答案句没被检索到，
  这里一定答错。这样 B0→B6 的差异就只归因于检索/切分，而不是模型抽风。
- ``ollama``：本地 8B（qwen3:8b），用于交互演示。
- ``deepseek``：OpenAI 兼容接口，用于终版数字（无 key 时自动跳过）。
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Protocol

import httpx

from ..chunking.structural import Chunk
from ..config import SETTINGS, Settings
from ..index.bm25 import tokenize
from ..retrieve.hybrid import canonical, expand

SYSTEM = (
    "你是一个严谨的文档问答助手。只能依据给定的证据回答，证据不足时必须拒答。"
    "输出必须是 JSON：{\"answer\":\"...\",\"citations\":[\"E1\"],\"refuse\":false}。"
    "答案里出现数字时，必须与证据完全一致，禁止换算或推断。"
)


@dataclass
class Evidence:
    eid: str
    chunk: Chunk

    @property
    def label(self) -> str:
        return f"{self.eid}（第 {self.chunk.page} 页 · {self.chunk.section or '无章节'}）"


@dataclass
class AnswerResult:
    text: str
    cited: list[str] = field(default_factory=list)
    refused: bool = False
    provider: str = ""
    latency_ms: int = 0
    raw: str = ""
    derived: bool = False          # 答案由计算得出（结果数字不在原文里）
    operands: list[float] = field(default_factory=list)


class Provider(Protocol):
    name: str

    def answer(self, question: str, evidences: list[Evidence],
               subject: str | None = None) -> AnswerResult: ...


def _sentences(text: str) -> list[str]:
    # 只按句末标点与换行切：表格行里用"；"分隔多列，
    # 按"；"切会把"2022：…" 的标签和后面的数字拆散（T5 就是这么丢的）
    parts = re.split(r"(?<=[。！？!?])|\n+", text)
    return [p.strip() for p in parts if p and p.strip()]


def _num(s: str) -> float | None:
    m = re.search(r"-?\d+(?:\.\d+)?", (s or "").replace(",", ""))
    return float(m.group(0)) if m else None


class ExtractiveProvider:
    """确定性基线：找与问题重合度最高的句子，找不到就拒答。"""

    name = "extractive"

    def __init__(self, min_overlap: float = 0.34):
        self.min_overlap = min_overlap

    def best_sentence(self, question: str, text: str) -> str:
        """给定一段块文本，挑出与问题最相关的一句（供"其他相关答案"列表用）。"""
        q = set(tokenize(canonical(question)))
        best, best_score = "", -1.0
        for sent in _sentences(text):
            if self.TITLE.match(sent) and len(sent) <= 40:
                continue
            sc = self._score(question, sent, q)
            if sc > best_score:
                best, best_score = sent, sc
        if best:
            return best
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        return (lines[0] if lines else text)[:200]

    def _score(self, question: str, sent: str, q_tokens: set[str] | None = None) -> float:
        # 用扩展后的词表打分：检索做了同义词扩展（手机号→手机），
        # 打分不扩展会出现"检索到了却判不相关"（实测手机号题被拒答）
        q = q_tokens if q_tokens is not None else set(tokenize(canonical(question)))
        s = set(tokenize(sent))
        if not q:
            return 0.0
        overlap = len(q & s) / len(q)
        # 短查询是"导航式"的（"目录""姓名""摘要"），文档里很可能只有同义词
        # （中文问"目录"、英文页写 Contents）。这类查询只要命中任一同义词
        # 就算相关，否则会被相似度阈值直接拒掉——用户想看的恰恰是那一段。
        if len(q) <= 2 and overlap == 0:
            syn = [t for t in tokenize(expand(question)) if t]
            low = sent.lower()
            if any(t in low for t in syn if len(t) >= 2):
                overlap = 0.6
        # 导航词（目录/摘要/参考文献/结论）出现在**段首**时几乎一定是那一页，
        # 否则"论文的目录"会被正文里提到"论文"的段落抢答（真实论文上踩到）
        head_norm = re.sub(r"\s+", "", sent[:24]).lower()
        for kw in ("目录", "目次", "摘要", "参考文献", "结论", "contents", "abstract"):
            if kw in question.lower() and head_norm.startswith(kw):
                # 目录里的"参考文献......67"也是以关键词开头，但它只是目录条目；
                # 用点线占比把它和真正的正文段落区分开（"目录"本身除外）
                if kw != "目录" and sent.count(".") > len(sent) * 0.2:
                    continue
                overlap += 0.45
                break
        nums_q = set(re.findall(r"\d+(?:\.\d+)?", question))
        nums_s = set(re.findall(r"\d+(?:\.\d+)?", sent))
        if nums_q:
            overlap += 0.35 * len(nums_q & nums_s) / len(nums_q)
        # 问"多少"却答了个百分比，是真实年报上最常见的错法（问营收、答增速）：
        # 没有问"率/比例"时，带 % 的句子降权
        if re.search(r"多少|几", question) and not self.RATE.search(question) and "%" in sent:
            overlap -= 0.25
        # "母公司"与"合并"是两套数，用户没点名母公司时优先合并口径
        if "母公司" in sent and "母公司" not in question:
            overlap -= 0.3
        # 表格行通常以指标名开头（"营业收入：2025 年=…"），
        # 叙述句里顺带提到指标不应抢答；限定"行式结构"避免误伤正文
        head = sent[:12]
        if "=" in sent and any(tok in head for tok in tokenize(question) if len(tok) >= 2):
            overlap += 0.2
        return overlap

    ROW_RE = re.compile(r"^(?P<label>[^：:]{1,24})[：:](?P<rest>.+)$")
    PAIR_RE = re.compile(r"(?P<col>[^=；;]+)=(?P<val>[^；;]+)")
    OP_MAX = re.compile(r"最高|最大|最多|最强")
    OP_MIN = re.compile(r"最低|最小|最少|最弱")
    MULTI = re.compile(r"分别|各是|两年|三年|对比|哪几年|各多少")
    ARITH = re.compile(r"增长|减少|下降|提升|提高|降低|增加|同比|环比|相差|差额")
    PREV = re.compile(r"上年|上期|同期|去年")
    RATE = re.compile(r"率|百分比|占比|比例|比重|%")
    UNIT = re.compile(r"亿元|万元|%|元")
    TITLE = re.compile(r"^(表|图|列名|（续上页）|\(续上页\))")

    def _rows(self, text: str) -> list[tuple[str, dict]]:
        """解析行式表格（"第三季度：营业收入（亿元）=12.4；上年同期（亿元）=12.4"）。

        注意：**按换行切，不能按句号/分号切**——行内用"；"分隔列，
        按"；"切会把"上年同期"那一列切丢，算术算子就永远拿不到同比数（踩过）。
        """
        rows = []
        for line in (text or "").split("\n"):
            line = line.strip()
            if not line or self.TITLE.match(line):
                continue
            m = self.ROW_RE.match(line)
            if not m:
                continue
            pairs = {mm.group("col").strip(): mm.group("val").strip()
                     for mm in self.PAIR_RE.finditer(m.group("rest"))}
            if pairs:
                rows.append((m.group("label").strip(), pairs))
        return rows

    def _pick_col(self, pairs: dict, question: str) -> str | None:
        qt = set(tokenize(question))
        best, best_s = None, 0
        for col in pairs:
            s = len(qt & set(tokenize(col)))
            if s > best_s:
                best, best_s = col, s
        return best or (next(iter(pairs)) if pairs else None)

    def _arith(self, question: str, evidences: list[Evidence]) -> AnswerResult | None:
        """表格算术：本期 − 上年同期（或增长率）。

        "第三季度营业收入相比上年同期增长了多少" 不是检索问题，是计算问题；
        抽取式如果只返回原始行，答案里就没有"增长额"，评测必然判错。

        但问"增长率/增速"时不能算：表里通常已经有这一列，再算一遍会得出
        "同比 12061656084%" 这种离谱数字（真实年报上真发生过）。
        """
        if not self.ARITH.search(question) or self.RATE.search(question):
            return None
        qt = set(tokenize(question))
        best = None
        for ev in evidences:
            for label, pairs in self._rows(ev.chunk.text):
                prev_col = next((c for c in pairs if self.PREV.search(c)), None)
                if not prev_col:
                    continue
                cur_col = self._pick_col({k: v for k, v in pairs.items() if k != prev_col}, question)
                if not cur_col:
                    continue
                a, b = _num(pairs[cur_col]), _num(pairs[prev_col])
                if a is None or b is None:
                    continue
                # 行标签命中问题（"第三季度"）优先，避免拿第一行凑数
                score = len(qt & set(tokenize(label))) * 3 + len(qt & set(tokenize(cur_col)))
                if best is None or score > best[0]:
                    best = (score, label, cur_col, prev_col, a, b, ev.eid)
        if best is None:
            return None
        _, label, cur_col, prev_col, a, b, eid = best
        if self.RATE.search(question) and b:
            text = f"{label}，同比 {(a - b) / abs(b) * 100:.1f}%"
        else:
            m = self.UNIT.search(cur_col)
            text = f"{label}，增长 {a - b:.1f}{m.group(0) if m else ''}"
        return AnswerResult(text=text, cited=[eid], provider=self.name,
                            derived=True, operands=[a, b])

    def _operate(self, question: str, evidences: list[Evidence]) -> AnswerResult | None:
        """最小表格算子：最高/最低 → 选列 → argmax/argmin。

        这是"结构化表格"才有的能力：B0 的定长切分没有"行-列"结构，
        这个算子拿不到输入，所以它天然成为 B2 的一个可测增量。
        """
        want_max = bool(self.OP_MAX.search(question))
        want_min = bool(self.OP_MIN.search(question))
        if not (want_max or want_min):
            return None

        # 先在"哪张表最相关"上做选择，再在选中的表内做 argmax/argmin。
        # 否则会拿"季度营收表"和"产品分类表"跨表比大小（实测确实会答成 25.48）。
        best_chunk = None
        best_hit = -1
        for ev in evidences:
            for _, pairs in self._rows(ev.chunk.text):
                col = self._pick_col(pairs, question)
                if not col:
                    continue
                score = len(set(tokenize(question)) & set(tokenize(col)))
                if score > best_hit:
                    best_hit, best_chunk = score, ev
                break
        if best_chunk is None:
            return None

        cands = []
        for label, pairs in self._rows(best_chunk.chunk.text):
            col = self._pick_col(pairs, question)
            if not col:
                continue
            v = _num(pairs[col])
            if v is None:
                continue
            cands.append((v, label, col, pairs[col]))
        if not cands:
            return None
        v, label, col, val = (max(cands, key=lambda x: x[0]) if want_max
                              else min(cands, key=lambda x: x[0]))
        return AnswerResult(text=f"{label}，{col}为 {val}", cited=[best_chunk.eid],
                            provider=self.name)

    def answer(self, question: str, evidences: list[Evidence],
               subject: str | None = None) -> AnswerResult:
        for fn in (self._arith, self._operate):
            r = fn(question, evidences)
            if r is not None:
                return r

        # 口径：UI 上选了就按选的来，没选就看问题里有没有"母公司"
        wants_parent = (subject == "parent") if subject else ("母公司" in question)

        best = (0.0, "", "", "")
        pool: list[tuple[float, str, str, set[str]]] = []
        q_tokens = set(tokenize(canonical(question)))
        for ev in evidences:
            # 图注通常很短，整段作为一个候选，避免被句号切碎只答半句
            cands = [ev.chunk.text] if ev.chunk.type == "figure" else _sentences(ev.chunk.text)
            for sent in cands:
                if ev.chunk.type != "figure" and self.TITLE.match(sent):
                    continue
                score = self._score(question, sent, q_tokens)
                # 口径标签：没点名"母公司"时，合并报表优先；点名了则反过来
                subj = ev.chunk.meta.get("subject")
                if subj == "parent":
                    score += 0.3 if wants_parent else -0.25
                elif subj == "consolidated":
                    score += 0.15 if not wants_parent else -0.1
                pool.append((score, sent, ev.eid, set(re.findall(r"\d+(?:\.\d+)?", sent))))
                if score > best[0]:
                    best = (score, sent, ev.eid, ev.chunk.id)
        if best[0] < self.min_overlap or not best[1]:
            return AnswerResult(text="根据当前文档无法确定该问题的答案。",
                                cited=[], refused=True, provider=self.name)

        text = best[1]
        # 目录页整块就是一个答案（"目录"问的是整张表），直接返回整块，
        # 只返回一行等于没回答
        ev_best = next((e for e in evidences if e.eid == best[2]), None)
        if ev_best is not None and ev_best.chunk.meta.get("toc"):
            return AnswerResult(text=ev_best.chunk.text.strip()[:4000],
                                cited=[best[2]], provider=self.name)
        # 命中的只是章节标题（"自我评价"）时，把下一行正文接上——
        # 简历/表单里标题和内容常常各占一行
        if len(text) <= 15 and "：" not in text and ":" not in text:
            ev = ev_best
            if ev is not None:
                lines = [ln.strip() for ln in ev.chunk.text.split("\n") if ln.strip()]
                try:
                    i = lines.index(text)
                except ValueError:
                    i = -1
                # 标题下面可能是日期行（"2025.11-2026.01"），跳过它取正文
                for j in range(i + 1, min(i + 4, len(lines))) if i >= 0 else []:
                    nxt = lines[j]
                    if re.fullmatch(r"[0-9.\-~至/年月]+", nxt) or len(nxt) < 4:
                        continue
                    text = f"{text}：{nxt}"
                    break
        # 多值问题（"分别/两年/对比"）要把多个数据行拼起来，
        # 只补一句会漏掉另一半数字（T5 之前 0% 就是这个原因）
        if self.MULTI.search(question):
            chosen = [best[1]]
            seen = set(re.findall(r"\d+(?:\.\d+)?", best[1]))
            for score, sent, _eid, nums in sorted(pool, key=lambda x: -x[0]):
                if sent in chosen or score <= 0:
                    continue
                if nums - seen:
                    chosen.append(sent)
                    seen |= nums
                    if len(chosen) >= 3:
                        break
            text = "；".join(chosen)
        return AnswerResult(text=text[:4000], cited=[best[2]], provider=self.name)


def _parse_json(text: str) -> dict:
    text = text.strip()
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


class OllamaProvider:
    name = "ollama"

    def __init__(self, settings: Settings = SETTINGS):
        self.settings = settings

    def _prompt(self, question: str, evidences: list[Evidence], subject: str | None = None) -> str:
        blocks = []
        for ev in evidences:
            body = ev.chunk.text[:1200]
            blocks.append(f"[{ev.eid}] 第{ev.chunk.page}页 · {ev.chunk.section}\n{body}")
        hint = {"parent": "（只采用母公司报表口径）",
                "consolidated": "（只采用合并报表口径）"}.get(subject or "", "")
        return (f"证据：\n\n" + "\n\n".join(blocks) +
                f"\n\n问题：{question}{hint}\n\n只输出 JSON。")

    def answer(self, question: str, evidences: list[Evidence],
               subject: str | None = None) -> AnswerResult:
        t0 = time.time()
        payload = {
            "model": self.settings.llm_model,
            "prompt": self._prompt(question, evidences, subject),
            "system": SYSTEM,
            "stream": False,
            "think": False,
            "format": "json",
            "options": {"temperature": self.settings.temperature,
                        "num_ctx": 4096,
                        "num_predict": self.settings.max_answer_tokens},
        }
        with httpx.Client(base_url=self.settings.ollama_url, timeout=300.0) as client:
            r = client.post("/api/generate", json=payload)
            r.raise_for_status()
            data = r.json()
        raw = data.get("response", "")
        obj = _parse_json(raw)
        return AnswerResult(
            text=(obj.get("answer") or "").strip() or raw[:300],
            cited=[c for c in obj.get("citations", []) if isinstance(c, str)],
            refused=bool(obj.get("refuse", False)),
            provider=self.name,
            latency_ms=int((time.time() - t0) * 1000),
            raw=raw,
        )


class DeepSeekProvider:
    name = "deepseek"

    def __init__(self, settings: Settings = SETTINGS):
        self.settings = settings

    def available(self) -> bool:
        return bool(self.settings.deepseek_api_key)

    def answer(self, question: str, evidences: list[Evidence],
               subject: str | None = None) -> AnswerResult:
        t0 = time.time()
        blocks = [f"[{ev.eid}] 第{ev.chunk.page}页 · {ev.chunk.section}\n{ev.chunk.text[:1200]}"
                  for ev in evidences]
        hint = {"parent": "（只采用母公司报表口径）",
                "consolidated": "（只采用合并报表口径）"}.get(subject or "", "")
        body = {
            "model": self.settings.deepseek_model,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": "证据：\n\n" + "\n\n".join(blocks) +
                 f"\n\n问题：{question}{hint}"},
            ],
            "temperature": self.settings.temperature,
            "response_format": {"type": "json_object"},
        }
        with httpx.Client(base_url=self.settings.deepseek_base_url, timeout=120.0) as client:
            r = client.post("/chat/completions", json=body,
                            headers={"Authorization": f"Bearer {self.settings.deepseek_api_key}"})
            r.raise_for_status()
            data = r.json()
        raw = data["choices"][0]["message"]["content"]
        obj = _parse_json(raw)
        return AnswerResult(
            text=(obj.get("answer") or "").strip() or raw[:300],
            cited=[c for c in obj.get("citations", []) if isinstance(c, str)],
            refused=bool(obj.get("refuse", False)),
            provider=self.name,
            latency_ms=int((time.time() - t0) * 1000),
            raw=raw,
        )


def get_provider(name: str, settings: Settings = SETTINGS) -> Provider:
    if name == "ollama":
        return OllamaProvider(settings)
    if name == "deepseek":
        return DeepSeekProvider(settings)
    return ExtractiveProvider()
