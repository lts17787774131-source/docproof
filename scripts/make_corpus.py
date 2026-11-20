"""语料生成器：造 3 份"长得像真年报"的中文 PDF + 对应的金标题集。

为什么要自己造语料：
- 公开年报体积大、版权与分发受限，且没有现成的"证据级"标注；
- 自造语料可以让**标准答案、证据页码、证据原文**三者严格对齐，
  这样算出来的 Recall@k / 引用准确率才是真的，而不是估计值。

注意：本语料是**合成数据**，只用于验证管线与指标，不是最终 benchmark。
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image as RLImage, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)
from PIL import Image, ImageDraw, ImageFont

FONT_PATH = r"C:\Windows\Fonts\simsun.ttc"
FONT_BOLD = r"C:\Windows\Fonts\simhei.ttf"
pdfmetrics.registerFont(TTFont("Song", FONT_PATH, subfontIndex=0))
pdfmetrics.registerFont(TTFont("Hei", FONT_BOLD))

H1 = ParagraphStyle("H1", fontName="Hei", fontSize=18, leading=26, spaceAfter=10)
H2 = ParagraphStyle("H2", fontName="Hei", fontSize=13, leading=20, spaceBefore=10, spaceAfter=6)
BODY = ParagraphStyle("BODY", fontName="Song", fontSize=10.5, leading=17, firstLineIndent=21)
CAP = ParagraphStyle("CAP", fontName="Song", fontSize=8.5, leading=12, textColor=colors.grey)

# ---------------------------------------------------------------- 事实库
# 每份文档的事实都写在这里，答案与证据片段一一对应。
DOCS = [
    {
        "doc_id": "hxkj_2023",
        "file": "华信科技2023年年度报告.pdf",
        "title": "华信科技股份有限公司 2023 年年度报告",
        "company": "华信科技股份有限公司",
        "auditor": "立信会计师事务所（特殊普通合伙）",
        "address": "上海市浦东新区张江路 88 号",
        "employees": "3,412",
        "rd": "4.82 亿元",
        "rd_ratio": "8.6%",
        "patents": "187 项",
        "risk": "原材料价格波动",
        "quarters": [("第一季度", "10.2", "8.1"), ("第二季度", "11.6", "10.9"),
                     ("第三季度", "12.4", "12.4"), ("第四季度", "14.8", "15.2")],
        "gross": [("2021", "31.2%"), ("2022", "32.8%"), ("2023", "34.1%")],
        "cf": [("2022", "5.6 亿元"), ("2023", "7.3 亿元")],
        "fig_note": "研发投入占营业收入比重由 2021 年的 6.9% 提升至 2023 年的 8.6%",
        "rd_series": [("2021", 6.9), ("2022", 7.8), ("2023", 8.6)],
        "hq": "上海",
    },
    {
        "doc_id": "hyzz_2022",
        "file": "恒远制造2022年度报告.pdf",
        "title": "恒远制造集团股份有限公司 2022 年年度报告",
        "company": "恒远制造集团股份有限公司",
        "auditor": "天健会计师事务所（特殊普通合伙）",
        "address": "江苏省苏州市工业园区星湖街 199 号",
        "employees": "5,806",
        "rd": "3.15 亿元",
        "rd_ratio": "5.2%",
        "patents": "96 项",
        "risk": "汇率波动风险",
        "quarters": [("第一季度", "18.6", "17.1"), ("第二季度", "19.4", "19.9"),
                     ("第三季度", "21.2", "20.8"), ("第四季度", "22.5", "23.6")],
        "gross": [("2020", "24.1%"), ("2021", "25.6%"), ("2022", "23.9%")],
        "cf": [("2021", "8.2 亿元"), ("2022", "6.9 亿元")],
        "fig_note": "2022 年境外收入占比达到 38.4%，较上年提升 4.1 个百分点",
        "rd_series": [("2020", 4.6), ("2021", 4.9), ("2022", 5.2)],
        "hq": "苏州",
    },
    {
        "doc_id": "ctjt_2023h",
        "file": "城投集团2023年半年度报告.pdf",
        "title": "城投建设集团有限公司 2023 年半年度报告",
        "company": "城投建设集团有限公司",
        "auditor": "大华会计师事务所（特殊普通合伙）",
        "address": "湖北省武汉市武昌区中南路 12 号",
        "employees": "2,197",
        "rd": "1.06 亿元",
        "rd_ratio": "2.4%",
        "patents": "38 项",
        "risk": "地方政府债务政策变化",
        "quarters": [("第一季度", "22.1", "20.4"), ("第二季度", "26.8", "24.9"),
                     ("第三季度", "0", "0"), ("第四季度", "0", "0")],
        "gross": [("2021", "18.4%"), ("2022", "17.2%"), ("2023H1", "16.5%")],
        "cf": [("2022H1", "3.1 亿元"), ("2023H1", "4.4 亿元")],
        "fig_note": "新签合同额 68.3 亿元，同比下降 3.2%",
        "rd_series": [("2021H1", 2.1), ("2022H1", 2.2), ("2023H1", 2.4)],
        "hq": "武汉",
    },
    {
        "doc_id": "zywl_2021",
        "file": "中远物流2021年度报告.pdf",
        "title": "中远物流股份有限公司 2021 年年度报告",
        "company": "中远物流股份有限公司",
        "auditor": "信永中和会计师事务所（特殊普通合伙）",
        "address": "天津市滨海新区新港路 66 号",
        "employees": "4,028",
        "rd": "2.37 亿元",
        "rd_ratio": "4.1%",
        "patents": "54 项",
        "risk": "国际油价与运价波动",
        "quarters": [("第一季度", "15.2", "14.8"), ("第二季度", "16.1", "15.3"),
                     ("第三季度", "17.8", "16.2"), ("第四季度", "18.4", "17.9")],
        "gross": [("2019", "19.8%"), ("2020", "21.4%"), ("2021", "22.7%")],
        "cf": [("2020", "6.1 亿元"), ("2021", "7.4 亿元")],
        "fig_note": "2021 年研发投入强度提升至 4.1%，较上年提高 0.6 个百分点",
        "rd_series": [("2019", 3.1), ("2020", 3.5), ("2021", 4.1)],
        "hq": "天津",
    },
    {
        "doc_id": "rfyy_2022",
        "file": "瑞丰医药2022年度报告.pdf",
        "title": "瑞丰医药科技股份有限公司 2022 年年度报告",
        "company": "瑞丰医药科技股份有限公司",
        "auditor": "普华永道中天会计师事务所（特殊普通合伙）",
        "address": "浙江省杭州市滨江区江南大道 588 号",
        "employees": "1,876",
        "rd": "6.42 亿元",
        "rd_ratio": "12.8%",
        "patents": "233 项",
        "risk": "药品集中带量采购降价",
        "quarters": [("第一季度", "9.8", "10.6"), ("第二季度", "10.4", "11.2"),
                     ("第三季度", "11.9", "10.1"), ("第四季度", "13.2", "12.4")],
        "gross": [("2020", "62.4%"), ("2021", "61.1%"), ("2022", "59.7%")],
        "cf": [("2021", "4.8 亿元"), ("2022", "5.9 亿元")],
        "fig_note": "2022 年研发投入强度达到 12.8%，为近三年最高",
        "rd_series": [("2020", 10.9), ("2021", 11.7), ("2022", 12.8)],
        "hq": "杭州",
    },
    {
        "doc_id": "lhny_2023",
        "file": "蓝海能源2023年度报告.pdf",
        "title": "蓝海能源集团有限公司 2023 年年度报告",
        "company": "蓝海能源集团有限公司",
        "auditor": "德勤华永会计师事务所（特殊普通合伙）",
        "address": "广东省深圳市南山区科技南路 18 号",
        "employees": "6,940",
        "rd": "1.88 亿元",
        "rd_ratio": "2.9%",
        "patents": "71 项",
        "risk": "新能源补贴退坡",
        "quarters": [("第一季度", "28.4", "31.2"), ("第二季度", "30.1", "32.6"),
                     ("第三季度", "33.7", "30.5"), ("第四季度", "36.2", "35.8")],
        "gross": [("2021", "21.6%"), ("2022", "20.3%"), ("2023", "19.1%")],
        "cf": [("2022", "12.6 亿元"), ("2023", "15.1 亿元")],
        "fig_note": "2023 年研发投入强度为 2.9%，较上年下降 0.3 个百分点",
        "rd_series": [("2021", 3.4), ("2022", 3.2), ("2023", 2.9)],
        "hq": "深圳",
    },
]


def _table(rows: list[list[str]], widths=None, header=True) -> Table:
    t = Table(rows, colWidths=widths, hAlign="LEFT")
    style = [
        ("FONTNAME", (0, 0), (-1, -1), "Song"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF7")),
                  ("FONTNAME", (0, 0), (-1, 0), "Hei")]
    t.setStyle(TableStyle(style))
    return t


def _chart_png(out_path: Path, series: list[tuple[str, float]], title: str) -> Path:
    """画一张真实的位图柱状图——必须是栅格图，否则 PDF 里没有 image block，
    我们的图表检测就拿不到 figure。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    W, H = 960, 420
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    f_title = ImageFont.truetype(FONT_BOLD, 26)
    f_lab = ImageFont.truetype(FONT_PATH, 22)

    L, R, T, B = 100, 50, 70, 70
    pw, ph = W - L - R, H - T - B
    vmax = max(v for _, v in series) * 1.3
    d.text((W // 2 - 90, 20), title, font=f_title, fill=(30, 30, 30))
    d.line([(L, T), (L, T + ph)], fill=(120, 120, 120), width=2)
    d.line([(L, T + ph), (L + pw, T + ph)], fill=(120, 120, 120), width=2)

    n = len(series)
    bw = pw / n * 0.45
    for i, (label, v) in enumerate(series):
        cx = L + pw * (i + 0.5) / n
        h = ph * v / vmax
        d.rectangle([cx - bw / 2, T + ph - h, cx + bw / 2, T + ph],
                    fill=(70, 120, 190), outline=(40, 70, 120))
        d.text((cx - 32, T + ph - h - 32), f"{v}", font=f_lab, fill=(20, 20, 20))
        d.text((cx - len(label) * 6, T + ph + 12), label, font=f_lab, fill=(60, 60, 60))
    img.save(out_path, dpi=(150, 150))
    return out_path


def build_pdf(doc: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    story = []
    d = doc

    # 封面
    story += [Spacer(1, 60 * mm),
              Paragraph(d["title"], ParagraphStyle("T", fontName="Hei", fontSize=24, leading=34, alignment=1)),
              Spacer(1, 12 * mm),
              Paragraph(f"股票代码：{600000 + len(d['company'])}　　披露日期：2024 年 4 月 18 日",
                        ParagraphStyle("C", fontName="Song", fontSize=11, alignment=1)),
              PageBreak()]

    # 第一节 公司概况
    story += [Paragraph("第一节 公司概况", H1),
              Paragraph(f"{d['company']}（以下简称“公司”）注册地址为{d['address']}，"
                        f"报告期末在职员工 {d['employees']} 人。公司主营业务为高端装备与工业软件的研发、生产与销售。", BODY),
              Paragraph("一、基本信息", H2),
              Paragraph(f"公司聘请的会计师事务所为{d['auditor']}，审计意见类型为标准无保留意见。", BODY),
              Paragraph("二、主要会计数据", H2),
              Paragraph("表 1 分季度主要财务数据", CAP),
              Spacer(1, 3)]
    rows = [["季度", "营业收入（亿元）", "上年同期（亿元）"]]
    for q, v, last in d["quarters"]:
        rows.append([q, v, last])
    story += [_table(rows, widths=[45 * mm, 55 * mm, 55 * mm]), Spacer(1, 6)]
    story += [Paragraph("注：以上数据未经审计。", CAP), PageBreak()]

    # 第二节 管理层讨论与分析（跨页表）
    story += [Paragraph("第二节 管理层讨论与分析", H1),
              Paragraph("一、经营情况讨论", H2),
              Paragraph(f"报告期内，公司实现营业收入稳步增长，第三季度单季增速较上半年明显加快。"
                        f"公司持续加大研发投入，报告期研发投入合计 {d['rd']}，"
                        f"占营业收入比重 {d['rd_ratio']}，累计获得授权专利 {d['patents']}。", BODY),
              Paragraph("二、毛利率变动", H2),
              Paragraph("表 2 近三年毛利率情况", CAP),
              Spacer(1, 3)]
    grows = [["年度", "毛利率"]]
    for y, v in d["gross"]:
        grows.append([y, v])
    story += [_table(grows, widths=[45 * mm, 45 * mm]), Spacer(1, 4)]

    # 故意让"现金流量"表跨页：先写好表头，再 PageBreak
    story += [Paragraph("三、现金流量情况", H2),
              Paragraph("表 3 经营活动产生的现金流量净额", CAP),
              Spacer(1, 3)]
    crows = [["年度", "经营活动现金流量净额"]]
    for y, v in d["cf"]:
        crows.append([y, v])
    half = max(2, len(crows) // 2)
    story += [_table(crows[:half], widths=[45 * mm, 65 * mm])]
    story += [PageBreak()]
    story += [Paragraph("（续上页）表 3 经营活动产生的现金流量净额", CAP), Spacer(1, 3),
              _table([crows[0]] + crows[half:], widths=[45 * mm, 65 * mm]), Spacer(1, 6)]

    chart = _chart_png(Path(tempfile.gettempdir()) / f"{d['doc_id']}_rd.png",
                       d["rd_series"], "研发投入强度（%）")
    story += [Paragraph("四、研发投入结构", H2),
              Spacer(1, 4),
              RLImage(str(chart), width=150 * mm, height=64 * mm),
              Paragraph(f"图 1 研发投入强度变化。{d['fig_note']}。", CAP),
              Spacer(1, 6),
              Paragraph("如图 1 所示，公司研发投入强度逐年提升，报告期末研发人员 612 人。", BODY)]

    total = sum(float(v) for _, v, _ in d["quarters"])
    story += [Paragraph("五、行业与竞争格局", H2),
              Paragraph("根据行业统计，报告期内国内高端装备市场规模约 1.2 万亿元，同比增长 6.8%；"
                        "行业集中度 CR5 为 38.6%，公司市场占有率 4.3%，较上年提升 0.5 个百分点。", BODY),
              Paragraph("公司主要竞争对手为三家头部企业。公司凭借研发投入强度与交付周期优势，"
                        "在高端客户中的复购率达到 82%，客户结构持续优化。", BODY),
              Paragraph("六、主要客户与供应商", H2),
              Paragraph("表 5 前五大客户销售情况", CAP),
              Spacer(1, 3)]
    cus = [["客户名称", "销售金额（亿元）", "占营业收入比例"]]
    for nm, sh in [("客户甲", 0.18), ("客户乙", 0.11), ("客户丙", 0.08),
                   ("客户丁", 0.06), ("客户戊", 0.04)]:
        cus.append([nm, f"{total * sh:.2f}", f"{sh * 100:.0f}%"])
    story += [_table(cus, widths=[55 * mm, 55 * mm, 45 * mm]), Spacer(1, 8),
              Paragraph("七、员工与子公司", H2),
              Paragraph(f"报告期末公司员工总数 {d['employees']} 人，其中研发人员占 17.9%。"
                        f"公司拥有 7 家子公司，主要分布于{d['hq']}、深圳、成都三地，"
                        f"子公司主要从事区域销售与技术服务。", BODY),
              Paragraph("八、未来展望", H2),
              Paragraph("公司将继续围绕主业加大研发投入，推进产能扩张与海外渠道建设，"
                        "同时加强应收账款管理，力争下一年度营业收入保持两位数增长。", BODY),
              PageBreak()]

    # 第三节 风险因素
    story += [Paragraph("第三节 风险因素", H1),
              Paragraph(f"一、{d['risk']}", H2),
              Paragraph(f"公司面临的主要风险包括{d['risk']}、下游需求不及预期、应收账款回收周期延长等。"
                        f"公司将通过套期保值、客户结构优化等措施加以应对。", BODY),
              PageBreak()]

    # 第四节 财务附注
    story += [Paragraph("第四节 财务报表附注", H1),
              Paragraph("一、合并范围", H2),
              Paragraph(f"本报告期合并范围包括母公司及 7 家子公司，注册地主要分布于{d['hq']}、深圳、成都三地。", BODY),
              Paragraph("二、营业收入明细", H2),
              Paragraph("表 4 营业收入按产品分类", CAP),
              Spacer(1, 3)]
    prod = [["产品类别", "收入（亿元）", "占比"]]
    total = sum(float(v) for _, v, _ in d["quarters"])
    for i, (name, share) in enumerate([("智能装备", 0.52), ("工业软件", 0.31), ("运维服务", 0.17)]):
        prod.append([name, f"{total * share:.2f}", f"{share * 100:.0f}%"])
    story += [_table(prod, widths=[55 * mm, 45 * mm, 35 * mm]),
              Spacer(1, 8),
              Paragraph("三、关联方交易", H2),
              Paragraph("报告期内公司不存在重大关联方交易，与控股股东之间的资金往来均为正常经营性往来。", BODY)]

    doc_tpl = SimpleDocTemplate(str(out_path), pagesize=A4,
                                leftMargin=25 * mm, rightMargin=25 * mm,
                                topMargin=22 * mm, bottomMargin=20 * mm,
                                title=d["title"])
    doc_tpl.build(story)


def build_questions(doc: dict) -> list[dict]:
    d = doc
    did = d["doc_id"]
    q: list[dict] = []

    def add(qtype, question, answer, snippets, pages=None):
        q.append({"qid": f"{did}-{len(q) + 1:03d}", "doc_id": did, "qtype": qtype,
                  "question": question, "answer": answer,
                  "evidence": [{"page": p if pages else 0, "text": s} for p, s in
                               zip(pages or [0] * len(snippets), snippets)],
                  "answerable": qtype != "T6"})

    # T1 单块事实
    add("T1", "公司聘请的会计师事务所是哪一家？", d["auditor"],
        [f"会计师事务所为{d['auditor']}"], [2])
    add("T1", "公司的注册地址在哪里？", d["address"], [d["address"]], [2])
    add("T1", "报告期末公司在职员工有多少人？", d["employees"], [f"在职员工 {d['employees']} 人"], [2])

    # T2 表格数值
    for qi, (qq, v, last) in enumerate(d["quarters"][:3]):
        if qq == "第三季度" and float(v) > 0:
            add("T2", "第三季度单季营业收入是多少亿元？", f"{v} 亿元",
                [f"{qq} {v}"], [3])
            add("T2", "第三季度营业收入相比上年同期增长了多少？",
                f"{float(v) - float(last):.1f} 亿元", [f"{qq} {v} {last}"], [3])
    add("T2", "报告期研发投入合计多少？", d["rd"], [f"研发投入合计 {d['rd']}"], [3])
    add("T2", "研发投入占营业收入的比重是多少？", d["rd_ratio"], [f"营业收入比重 {d['rd_ratio']}"], [3])

    # T3 表格推理
    best = max(d["quarters"][:4], key=lambda x: float(x[1]))
    add("T3", "哪个季度的营业收入最高？", best[0], [f"{best[0]} {best[1]}"], [3])
    gy = max(d["gross"], key=lambda x: float(x[1].rstrip("%")))
    add("T3", "哪一年的毛利率最高？", gy[0], [f"{gy[0]} {gy[1]}"], [4])

    # T4 图表
    add("T4", "图 1 反映的研发投入强度变化是怎样的？", d["fig_note"], [d["fig_note"]], [5])

    # T5 跨页
    add("T5", "经营活动产生的现金流量净额最近两年分别是多少？",
        "；".join(f"{y} 年 {v}" for y, v in d["cf"]), [f"{y} {v}" for y, v in d["cf"]], [4, 5])

    # T6 不可回答
    add("T6", "公司 2025 年的营业收入是多少？", "NOT_IN_DOC", [])
    add("T6", "公司董事长的姓名是什么？", "NOT_IN_DOC", [])
    return q


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    raw = root / "data" / "raw"
    evald = root / "eval" / "datasets"
    raw.mkdir(parents=True, exist_ok=True)
    evald.mkdir(parents=True, exist_ok=True)

    all_q: list[dict] = []
    manifest = []
    for d in DOCS:
        p = raw / d["file"]
        build_pdf(d, p)
        qs = build_questions(d)
        all_q.extend(qs)
        manifest.append({"doc_id": d["doc_id"], "file": str(p), "n_questions": len(qs)})
        print(f"[corpus] {p.name}: {p.stat().st_size // 1024} KB, {len(qs)} questions")

    (evald / "gold.jsonl").write_text(
        "\n".join(json.dumps(q, ensure_ascii=False) for q in all_q), encoding="utf-8")
    (evald / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1),
                                         encoding="utf-8")
    print(f"[corpus] total questions: {len(all_q)} -> {evald / 'gold.jsonl'}")


if __name__ == "__main__":
    main()
