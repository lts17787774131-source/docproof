"""集中配置：路径、模型、检索与切分参数.

所有可调参数集中在这里，评测脚本通过 ``override`` 生成不同实验配置，
避免"参数散落在代码里导致消融不可复现"。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    root: Path = ROOT
    data_dir: Path = ROOT / "data"
    raw_dir: Path = ROOT / "data" / "raw"
    pages_dir: Path = ROOT / "data" / "pages"
    parsed_dir: Path = ROOT / "data" / "parsed"
    db_path: Path = ROOT / "data" / "docproof.db"
    eval_dir: Path = ROOT / "eval"

    # 模型服务
    ollama_url: str = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
    embed_model: str = os.getenv("DOCPROOF_EMBED_MODEL", "bge-m3")
    llm_model: str = os.getenv("DOCPROOF_LLM_MODEL", "qwen3:8b")
    # DeepSeek（终版评测用；无 key 时自动跳过）
    deepseek_base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    deepseek_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")

    # 解析
    page_dpi: int = 150
    text_layer_min_chars: int = 40
    max_pages: int = 0  # >0 时只解析前 N 页（大文档冒烟用）

    # 切分
    child_size: int = 240
    parent_size: int = 1600
    fixed_size: int = 512

    # 检索
    top_k: int = 8
    recall_k: int = 5
    rrf_k: int = 60
    min_score: float = 0.12  # 低于该分视为"证据不足"→ 拒答

    # 生成
    max_answer_tokens: int = 256
    temperature: float = 0.0

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.raw_dir, self.pages_dir, self.parsed_dir,
                  self.eval_dir / "datasets", self.eval_dir / "reports"):
            p.mkdir(parents=True, exist_ok=True)


SETTINGS = Settings()


def override(base: Settings, **kwargs) -> Settings:
    """返回一个替换了部分字段的新配置（frozen dataclass 不可变）。"""
    return replace(base, **kwargs)
