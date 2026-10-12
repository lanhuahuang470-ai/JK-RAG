"""
DashScope Rerank 精排器（gte-rerank-v2）

为什么需要 Rerank：
- 召回阶段（向量/BM25）是双塔打分，query 与文档独立编码，精度有限
- Rerank 是交叉编码（cross-encoder），query 与文档一起进模型，相关性判断更准
- 代价是慢，所以只对召回的小集合（Top-N≈20）精排，再截断到最终 Top-K
"""
from __future__ import annotations

from typing import List, Tuple

import dashscope
from dashscope import TextReRank
from langchain_core.documents import Document

from config.settings import settings
from src.utils import get_logger

logger = get_logger(__name__)


class DashScopeReranker:
    """调用 DashScope gte-rerank 模型对候选文档精排"""

    def __init__(self, model: str = settings.rerank_model) -> None:
        self.model = model
        dashscope.api_key = settings.dashscope_api_key

    def rerank(
        self, query: str, docs: List[Document], top_n: int
    ) -> List[Tuple[Document, float]]:
        """
        精排并截断到 top_n

        Returns:
            [(Document, relevance_score)]：relevance_score 为 0~1，
            语义是"模型判定的相关度"，可直接作为检索置信度展示
        """
        if not docs:
            return []
        try:
            resp = TextReRank.call(
                model=self.model,
                query=query,
                documents=[d.page_content for d in docs],
                top_n=top_n,
            )
            if resp.status_code != 200:
                raise RuntimeError(f"Rerank API 错误：{resp.code} {resp.message}")
            results = resp.output["results"]  # [{"index": i, "relevance_score": s}, ...]
            return [
                (docs[r["index"]], float(r["relevance_score"])) for r in results
            ]
        except Exception as e:
            # 精排是增强而非必需品：失败时回退召回顺序，保证主链路可用
            logger.warning(f"Rerank 失败，回退召回顺序：{e}")
            return [(d, 0.0) for d in docs[:top_n]]
