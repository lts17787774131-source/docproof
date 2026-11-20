# DocProof v0.1 发布说明

> 首个可用版本：解析 → 索引 → 检索 → 生成/拒答 → 引用高亮，全链路可跑、可评测。

## 亮点

- **可溯源**：每个答案带 `[En]` 引用，点一下定位到原文页并画出 bbox；引用可复制整块原文。
- **多候选**：一次给出 top-5 相关答案，逐条可定位/复制，问"名字""目录"这类题不用反复追问。
- **敢拒答**：数字闸门、未来年份闸门、覆盖度校验三重把关；拒答时仍给候选。
- **真实口径**：识别"合并/母公司"报表并打标签，同一问题可切换口径对照。

## 评测结果

| 集合 | 结果 |
|---|---|
| 合成集（76 题，单变量消融） | **46.9% → 98.4%**（+51.6pp，95% CI [39.1, 64.1]，McNemar p<1e-4，33 胜 0 负） |
| 真实年报口语集（15 题，2 份共 438 页） | **67% → 87%** |
| 误拒率 / 不可回答题拒答率 | 2% / 100% |
| 解析规模 | 161 页 / 220 表、277 页 / 293 表；277 页约 75 秒建完索引 |

## 主要能力

| 模块 | 内容 |
|---|---|
| 解析 | PyMuPDF 文本层 + 表格（合并单元格还原）+ 图像块 → DocIR（page/bbox/section） |
| 切分 | 结构优先：表格整表、表头随行、续表合并、父子块、目录块 |
| 检索 | 自研 BM25（jieba + 数字特判）+ bge-m3 向量 + RRF 融合 |
| 生成 | extractive（评测用）/ Ollama qwen3:8b（本地）/ DeepSeek API（云端），Provider 可切换 |
| 服务 | FastAPI + 单页前端（口径切换、bbox 高亮、多候选、索引增删重建、深链 `?doc=&q=`） |
| 评测 | 76 题合成集 + 15 题真实口语集；bootstrap 置信区间 + McNemar 配对检验 |

## 已知限制

1. 扫描件不支持 OCR（只处理有文本层的 PDF）。
2. 跨页表头会丢（部分列名显示为"列5"）。
3. 图表题答的是图注文本，尚未接入 VLM 直读。
4. 建索引是同步的，大文档会阻塞上传（V1.5 计划改队列）。
5. 合成集 98.4% 是模板题上限，真实口语集 87% 才是真实水平。

## 快速开始

```powershell
git clone https://github.com/lts17787774131-source/docproof.git
cd docproof
python -m venv .venv && .\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\make_corpus.py       # 生成 6 份合成年报 + 76 题
.\.venv\Scripts\python.exe scripts\build_index.py --reset
.\.venv\Scripts\python.exe src\docproof\eval\runner.py   # 复现消融
$env:PYTHONPATH="$PWD\src"; .\.venv\Scripts\python.exe -m uvicorn docproof.api.app:app --port 7860
```

前提：本地 Ollama 已 `ollama pull bge-m3` 与 `ollama pull qwen3:8b`。

## 完整评测

见 [docs/eval-report-v0.1.md](eval-report-v0.1.md)。
