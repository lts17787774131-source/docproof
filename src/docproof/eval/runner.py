"""消融实验 runner：同一批题、同一生成器，只改一个变量。

设计要点（面试会被追问的都在这里）：
- 生成器固定为 extractive：B0→B3 的差异只能来自解析/切分/检索；
- 每个 config 独立重建 chunk 与索引，避免缓存串味；
- 每题保存 prediction（答案、引用、top-k、耗时），支持事后错误归因；
- 报 point estimate + bootstrap 置信区间 + McNemar。
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from docproof.chunking.structural import build_chunks  # noqa: E402
from docproof.config import SETTINGS  # noqa: E402
from docproof.eval.metrics import (answer_match, bootstrap_ci, citation_precision,  # noqa: E402
                                   evidence_in, mcnemar)
from docproof.generate.answer import Answerer  # noqa: E402
from docproof.generate.providers import ExtractiveProvider  # noqa: E402
from docproof.index.dense import OllamaEmbedder  # noqa: E402
from docproof.ingest.pipeline import parse_pdf  # noqa: E402
from docproof.retrieve.hybrid import Retriever  # noqa: E402

CONFIGS = [
    {"id": "B0", "name": "定长512 + BM25", "mode": "fixed",
     "table_view": "flat", "dense": False, "boost": False},
    {"id": "B1", "name": "结构切分 + BM25", "mode": "structural",
     "table_view": "flat", "dense": False, "boost": False},
    {"id": "B2", "name": "+ 表头随行序列化", "mode": "structural",
     "table_view": "rows", "dense": False, "boost": False},
    {"id": "B3", "name": "+ 向量混合 + RRF", "mode": "structural",
     "table_view": "rows", "dense": True, "boost": False},
    {"id": "B4", "name": "+ 题型路由（表格优先）", "mode": "structural",
     "table_view": "rows", "dense": False, "boost": True},
]


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, int(round(0.95 * (len(s) - 1))))
    return s[idx]


def load_gold() -> list[dict]:
    path = SETTINGS.eval_dir / "datasets" / "gold.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_docs() -> dict[str, object]:
    manifest = json.loads((SETTINGS.eval_dir / "datasets" / "manifest.json").read_text(encoding="utf-8"))
    docs = {}
    for m in manifest:
        docs[m["doc_id"]] = parse_pdf(m["file"], SETTINGS)
    return docs


def run_config(cfg: dict, docs: dict, questions: list[dict], embedder) -> dict:
    chunks_by_doc: dict[str, list] = {}
    for did, doc in docs.items():
        chunks_by_doc[did] = build_chunks(doc, mode=cfg["mode"], settings=SETTINGS,
                                          table_view=cfg["table_view"])

    embeddings: dict[str, tuple] = {}
    if cfg["dense"]:
        for did, chunks in chunks_by_doc.items():
            vecs = embedder.embed([c.text for c in chunks])
            embeddings[did] = ([c.id for c in chunks], vecs)

    answerer = Answerer(Retriever([], None), ExtractiveProvider(), SETTINGS)
    retrievers: dict[str, Retriever] = {}
    for did, chunks in chunks_by_doc.items():
        retrievers[did] = Retriever(chunks, embedder=embedder if cfg["dense"] else None,
                                    embeddings=embeddings.get(did))

    rows = []
    for q in questions:
        did = q["doc_id"]
        retriever = retrievers[did]
        answerer.retriever = retriever

        hits = retriever.search(q["question"], top_k=SETTINGS.top_k,
                                use_dense=cfg["dense"], rrf_k=SETTINGS.rrf_k,
                                type_boost=cfg["boost"])
        snippets = [e["text"] for e in q.get("evidence", []) if e.get("text")]
        def _hit_at(h) -> bool:
            return any(evidence_in(h.chunk.text, s) for s in snippets)

        first_rank = next((i + 1 for i, h in enumerate(hits) if _hit_at(h)), 0)
        recall_hit = first_rank > 0 and first_rank <= SETTINGS.recall_k
        recall3 = first_rank > 0 and first_rank <= 3

        payload = answerer.ask(q["question"], doc_id=did, use_dense=cfg["dense"],
                               type_boost=cfg["boost"],
                               doc_year=docs[did].meta.get("report_year"))
        cited = [{"snippet": c.snippet} for c in payload.citations]
        c_ok, c_tot = citation_precision(cited, snippets)

        is_t6 = q["qtype"] == "T6"
        ans_ok = (not payload.refused) and answer_match(payload.answer, q["answer"])
        rows.append({
            "qid": q["qid"], "doc_id": did, "qtype": q["qtype"], "question": q["question"],
            "gold": q["answer"], "pred": payload.answer, "refused": payload.refused,
            "refusal_reason": payload.refusal_reason,
            "correct": bool((payload.refused and is_t6) or ans_ok),
            "answer_correct": bool(ans_ok),
            "recall5": recall_hit, "recall3": recall3,
            "mrr": (1.0 / first_rank) if first_rank else 0.0,
            "cite_ok": c_ok, "cite_tot": c_tot,
            "latency_ms": payload.latency_ms,
            "top1": hits[0].chunk.text[:120] if hits else "",
        })

    answerable = [r for r in rows if r["qtype"] != "T6"]
    unanswerable = [r for r in rows if r["qtype"] == "T6"]
    acc = sum(r["answer_correct"] for r in answerable) / max(len(answerable), 1)
    refuse_acc = sum(r["refused"] for r in unanswerable) / max(len(unanswerable), 1)
    false_refusal = sum(r["refused"] for r in answerable) / max(len(answerable), 1)
    cite_ok = sum(r["cite_ok"] for r in answerable)
    cite_tot = sum(r["cite_tot"] for r in answerable)

    by_type = {}
    for t in sorted({r["qtype"] for r in answerable}):
        sub = [r for r in answerable if r["qtype"] == t]
        by_type[t] = round(sum(r["answer_correct"] for r in sub) / max(len(sub), 1), 4)

    return {
        "config": cfg,
        "n_questions": len(rows),
        "acc": round(acc, 4),
        "recall5": round(sum(r["recall5"] for r in answerable) / max(len(answerable), 1), 4),
        "recall3": round(sum(r["recall3"] for r in answerable) / max(len(answerable), 1), 4),
        "mrr": round(sum(r["mrr"] for r in answerable) / max(len(answerable), 1), 4),
        "citation_precision": round(cite_ok / cite_tot, 4) if cite_tot else None,
        "refusal_acc": round(refuse_acc, 4),
        "false_refusal": round(false_refusal, 4),
        "p95_latency_ms": p95([r["latency_ms"] for r in rows]),
        "by_type": by_type,
        "rows": rows,
    }


def main() -> None:
    SETTINGS.ensure_dirs()
    questions = load_gold()
    docs = load_docs()
    embedder = OllamaEmbedder(SETTINGS.ollama_url, SETTINGS.embed_model,
                              cache_path=SETTINGS.data_dir / "embed_cache.db")

    results = []
    for cfg in CONFIGS:
        t0 = time.time()
        r = run_config(cfg, docs, questions, embedder)
        r["elapsed_s"] = round(time.time() - t0, 1)
        results.append(r)
        print(f"[{cfg['id']}] acc={r['acc']:.3f} recall@5={r['recall5']:.3f} "
              f"citeP={r['citation_precision']} false_refusal={r['false_refusal']:.3f} "
              f"({r['elapsed_s']}s)")

    # 配对对比：每个 config 都相对 B0 做一次，找出真正的最优组
    comparisons = []
    base = {r["qid"]: r for r in results[0]["rows"]}
    for cand in results[1:]:
        cur = {r["qid"]: r for r in cand["rows"]}
        pairs, b, c = [], 0, 0
        for qid, r0 in base.items():
            r1 = cur[qid]
            if r0["qtype"] == "T6":
                continue
            pairs.append((int(r0["answer_correct"]), int(r1["answer_correct"])))
            if not r0["answer_correct"] and r1["answer_correct"]:
                b += 1
            elif r0["answer_correct"] and not r1["answer_correct"]:
                c += 1
        delta, lo, hi = bootstrap_ci(pairs)
        comparisons.append({
            "baseline": results[0]["config"]["id"], "candidate": cand["config"]["id"],
            "delta": round(delta, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "mcnemar_p": round(mcnemar(b, c), 6), "discordant_b": b, "discordant_c": c,
            "n_pairs": len(pairs),
        })
    best = max(results, key=lambda r: r["acc"])
    comparison = next((c for c in comparisons if c["candidate"] == best["config"]["id"]),
                      comparisons[-1] if comparisons else {})

    out_dir = SETTINGS.eval_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ablation.json").write_text(
        json.dumps({"results": [{k: v for k, v in r.items() if k != "rows"} for r in results],
                    "comparison": comparison, "comparisons": comparisons},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    for r in results:
        (out_dir / f"predictions_{r['config']['id']}.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in r["rows"]), encoding="utf-8")

    lines = ["# 消融实验报告（自动生成）", "",
             f"- 语料：{len(docs)} 份合成年报，题目 {len(questions)} 道",
             f"- 生成器：extractive（固定），embedding：{SETTINGS.embed_model}",
             "- 指标：Answer Acc 只在可答题（T1-T5）上统计；T6 看拒答率", "",
             "| 组 | 配置 | Acc | T1 | T2 | T3 | T4 | T5 | R@3 | R@5 | MRR | CiteP | 误拒 | p95(ms) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        bt = r["by_type"]
        lines.append(
            f"| {r['config']['id']} | {r['config']['name']} | {r['acc']*100:.1f}% | "
            f"{bt.get('T1',0)*100:.0f}% | {bt.get('T2',0)*100:.0f}% | {bt.get('T3',0)*100:.0f}% | "
            f"{bt.get('T4',0)*100:.0f}% | {bt.get('T5',0)*100:.0f}% | {r['recall3']*100:.0f}% | "
            f"{r['recall5']*100:.0f}% | {r['mrr']:.2f} | "
            f"{(r['citation_precision'] or 0)*100:.0f}% | {r['false_refusal']*100:.0f}% | "
            f"{r['p95_latency_ms']} |")
    lines += ["", f"## 配对检验（B0 vs {best['config']['id']}，最优组）", "",
              f"- ΔAcc = {comparison.get('delta',0)*100:+.1f}pp，"
              f"95% CI [{comparison.get('ci95',[0,0])[0]*100:+.1f}, {comparison.get('ci95',[0,0])[1]*100:+.1f}]，"
              f"n={comparison.get('n_pairs',0)}",
              f"- McNemar：b={comparison.get('discordant_b',0)}（基线错→新对），"
              f"c={comparison.get('discordant_c',0)}（基线对→新错），p={comparison.get('mcnemar_p',1):.4f}",
              "", "## 全部对比（相对 B0）", "",
              "| 组 | ΔAcc | 95% CI | McNemar p |", "|---|---|---|---|"]
    for c in comparisons:
        lines.append(f"| {c['candidate']} | {c['delta']*100:+.1f}pp | "
                     f"[{c['ci95'][0]*100:+.1f}, {c['ci95'][1]*100:+.1f}] | {c['mcnemar_p']:.4f} |")
    lines += ["", f"> 注意：{len(questions)} 道题（可答 {comparison.get('n_pairs',0)} 道）"
                  "只够做趋势判断，置信区间必然很宽；正式结论需要 300 题。"]
    (out_dir / "ablation.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
