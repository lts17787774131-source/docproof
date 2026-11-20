"""真实年报验证：从巨潮资讯拉 2 份公开年报，用同一条管线解析并试问。

目的不是出分数（真实年报没有金标），而是回答三个问题：
1) 解析器在真实版式上会不会崩？表格/图能识别多少？
2) 索引时间与体积在可接受范围吗？
3) 真实问题（非模板）上，检索和拒答还成立吗？

用法：python scripts/fetch_real_reports.py --n 2 --max-pages 40
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from docproof.config import SETTINGS, override  # noqa: E402
from docproof.ingest.pipeline import parse_pdf  # noqa: E402

H = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Referer": "http://www.cninfo.com.cn/new/commonUrl?url=disclosure/list/notice",
}
BAD = re.compile(r"摘要|英文|问询|回复|公告|已取消|债券|社会责任|内部控制|审计报告")


def query(limit: int = 40) -> list[dict]:
    body = {"pageNum": 1, "pageSize": limit, "column": "szse", "tabName": "fulltext",
            "plate": "", "stock": "", "searchkey": "",
            "secdate": "2024-03-01~2024-04-30", "tDate": "", "type": "",
            "category": "category_ndbg_szsh", "isHLtitle": "true"}
    with httpx.Client(timeout=30.0, headers=H, follow_redirects=True) as c:
        r = c.post("http://www.cninfo.com.cn/new/hisAnnouncement/query", data=body)
        r.raise_for_status()
        data = r.json()
    out = []
    for a in data.get("announcements") or []:
        title = re.sub(r"<[^>]+>", "", a.get("announcementTitle") or "")
        if "年度报告" not in title or BAD.search(title):
            continue
        out.append({"sec": a.get("secName"), "title": title,
                    "url": "http://static.cninfo.com.cn/" + (a.get("adjunctUrl") or "")})
    return out


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 50_000:
        return dest
    with httpx.Client(timeout=120.0, headers=H, follow_redirects=True) as c:
        r = c.get(url)
        r.raise_for_status()
        dest.write_bytes(r.content)
    return dest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--max-pages", type=int, default=40)
    args = ap.parse_args()

    cands = query()
    print(f"[cninfo] 候选年报 {len(cands)} 份，取前 {args.n} 份")
    raw = ROOT / "data" / "raw_real"
    cfg = override(SETTINGS, max_pages=args.max_pages, raw_dir=raw)
    cfg.ensure_dirs()

    report = []
    for item in cands[: args.n]:
        ann_id = Path(item["url"]).stem  # finalpage/.../12256xxxx.PDF -> 唯一编号
        safe = re.sub(r"[^\w\u4e00-\u9fff]+", "_", f"{item['sec']}_{ann_id}")[:70] + ".pdf"
        dest = download(item["url"], raw / safe)
        size_mb = dest.stat().st_size / 1e6
        doc = parse_pdf(dest, cfg)
        print(f"[parse] {item['sec']} | {item['title'][:40]} | {size_mb:.1f}MB "
              f"pages(截断)={doc.meta['n_pages']} blocks={doc.meta['n_blocks']} "
              f"tables={doc.meta['n_tables']} figs={doc.meta['n_figures']} "
              f"scanned={doc.meta['scanned_pages']} {doc.meta['parse_seconds']}s")
        report.append({"sec": item["sec"], "title": item["title"], "file": str(dest),
                       "size_mb": round(size_mb, 1), **{k: doc.meta[k] for k in
                       ("n_pages", "n_blocks", "n_tables", "n_figures", "scanned_pages")}})
        doc.save(cfg.parsed_dir / f"{doc.doc_id}.json")

    (ROOT / "eval" / "reports" / "real_reports.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[done] -> eval/reports/real_reports.json")


if __name__ == "__main__":
    main()
