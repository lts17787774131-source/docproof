"""纯 Python BM25（不依赖 rank_bm25），中文用 jieba 分词 + 数字/百分比特判。"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

import jieba

jieba.setLogLevel(20)

STOP = {
    "的", "了", "和", "与", "及", "在", "是", "为", "对", "中", "上", "下", "有", "无", "不",
    "也", "就", "都", "而", "等", "我", "你", "他", "她", "它", "这", "那", "什么", "多少",
    "哪", "如何", "请问", "请", "帮", "一下", "是多少", "分别为", "分别", "以及", "其中",
}
NUM_RE = re.compile(r"\d+(?:\.\d+)?%?")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("％", "%").replace("．", ".")
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    text = normalize(text)
    # "手  机" 这种中文之间的排版空格会把词拆散（jieba 会切成"手""机"），
    # 先去掉中日韩字符之间的空格；英文/数字之间的空格保留
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
    tokens: list[str] = []
    for w in jieba.lcut(text):
        w = w.strip().lower()
        if not w or w in STOP:
            continue
        if not re.search(r"[\w\u4e00-\u9fff]", w):
            continue
        tokens.append(w)
    # 数字单独进倒排：财报题的检索信号主要来自数字与年份
    tokens.extend(NUM_RE.findall(text))
    return tokens


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [tokenize(d) for d in docs]
        self.n = len(self.docs)
        self.doc_len = [len(d) for d in self.docs]
        self.avgdl = (sum(self.doc_len) / self.n) if self.n else 0.0
        self.tf: list[Counter] = [Counter(d) for d in self.docs]
        self.df: Counter = Counter()
        for d in self.docs:
            for t in set(d):
                self.df[t] += 1

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        q = tokenize(query)
        if not q or not self.n:
            return []
        scores = [0.0] * self.n
        for term in q:
            idf = self._idf(term)
            if idf <= 0:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(term, 0)
                if not f:
                    continue
                dl = self.doc_len[i] or 1
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                scores[i] += idf * f * (self.k1 + 1) / denom
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(i, s) for i, s in ranked[:k] if s > 0]
