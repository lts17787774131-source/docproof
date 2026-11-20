"""评测指标：答案准确率、Recall@k、引用准确率、拒答准确率.

口径（写进报告首页，防止漂移）：
- Answer Acc：数值题要求"标准答案里的数字全部出现在预测中"；文本题要求包含或 80% 词覆盖。
- Recall@k：金标证据片段是否出现在 top-k 块中（数字全中 + 70% 词覆盖）。
- Citation P：引用块是否真的包含金标证据。
- Refusal Acc：T6 题被正确拒答的比例；False Refusal：可答题被误拒的比例。
"""
from __future__ import annotations

import random
import re
import unicodedata

NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = s.replace("％", "%").replace("．", ".")
    return re.sub(r"\s+", "", s).lower()


def nums(s: str) -> set[str]:
    out = set()
    for n in NUM_RE.findall(unicodedata.normalize("NFKC", s or "")):
        try:
            out.add(f"{float(n):g}")
        except ValueError:
            out.add(n)
    return out


def _tokens(s: str) -> set[str]:
    """中文用**字符 bigram**：整段连续汉字会被 greedy 成一个 token，
    那样"覆盖率"退化成子串匹配，指标就失去区分度（踩过这个坑）。"""
    s = norm(s)
    toks: set[str] = set()
    for run in re.findall(r"[\u4e00-\u9fff]{2,}", s):
        toks |= {run[i:i + 2] for i in range(len(run) - 1)}
    toks |= set(re.findall(r"[a-z]{2,}", s))
    toks |= set(re.findall(r"\d+(?:\.\d+)?", s))
    return toks


def answer_match(pred: str, gold: str) -> bool:
    if not pred:
        return False
    g_nums = nums(gold)
    if g_nums:
        return g_nums <= nums(pred)
    gp, gg = norm(pred), norm(gold)
    if gg and gg in gp:
        return True
    gt, pt = _tokens(gold), _tokens(pred)
    return bool(gt) and len(gt & pt) / len(gt) >= 0.8


def evidence_in(text: str, snippet: str) -> bool:
    if not snippet:
        return False
    nt, ns = norm(text), norm(snippet)
    if ns and ns in nt:
        return True
    s_nums = nums(snippet)
    if s_nums and not s_nums <= nums(text):
        return False
    st, tt = _tokens(snippet), _tokens(text)
    if not st:
        return False
    return len(st & tt) / len(st) >= 0.8


def citation_precision(cited: list[dict], snippets: list[str]) -> tuple[int, int]:
    """返回 (正确引用数, 引用总数)。"""
    if not cited:
        return 0, 0
    ok = 0
    for c in cited:
        if any(evidence_in(c.get("snippet", ""), s) for s in snippets):
            ok += 1
    return ok, len(cited)


def bootstrap_ci(pairs: list[tuple[int, int]], n_boot: int = 2000, alpha: float = 0.05):
    """配对 bootstrap：pairs = [(baseline_correct, new_correct), ...]。"""
    if not pairs:
        return (0.0, 0.0, 0.0)
    rng = random.Random(42)
    diffs = []
    n = len(pairs)
    for _ in range(n_boot):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        base = sum(a for a, _ in sample) / n
        new = sum(b for _, b in sample) / n
        diffs.append(new - base)
    diffs.sort()
    point = sum(b for _, b in pairs) / n - sum(a for a, _ in pairs) / n
    lo = diffs[int(n_boot * alpha / 2)]
    hi = diffs[int(n_boot * (1 - alpha / 2))]
    return (point, lo, hi)


def mcnemar(b: int, c: int) -> float:
    """配对检验的粗略 p 值（连续性校正卡方），b/c 为两方向不一致的样本数。"""
    if b + c == 0:
        return 1.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    # 卡方 1 自由度 -> p 值近似
    import math
    return math.erfc(math.sqrt(chi2 / 2))
