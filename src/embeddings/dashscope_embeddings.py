"""
Qwen3-Embedding 嵌入模型封装
通过 DashScope OpenAI 兼容接口调用 text-embedding-v4（Qwen3-Embedding）
"""
from __future__ import annotations

import os
from typing import List

from langchain_core.embeddings import Embeddings
from openai import OpenAI

from config.settings import settings
from src.utils import get_logger

logger = get_logger(__name__)

# DashScope OpenAI 兼容端点
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class DashScopeEmbeddings(Embeddings):
    """
    Qwen3-Embedding（text-embedding-v4）LangChain 适配器

    通过 DashScope 的 OpenAI 兼容接口调用，支持：
    - 批量嵌入（embed_documents）
    - 单条查询嵌入（embed_query）

    注意：text-embedding-v4 单次最多 6 条文本（官方限制），
    本类自动做批次分割。
    """

    BATCH_SIZE = 6  # DashScope 单次批量上限

    def __init__(
        self,
        model: str = settings.embedding_model,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        _key = api_key or settings.dashscope_api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        if not _key:
            raise ValueError("DASHSCOPE_API_KEY 未配置，无法初始化嵌入模型。")

        self._client = OpenAI(
            api_key=_key,
            base_url=DASHSCOPE_BASE_URL,
        )
        logger.info(f"DashScopeEmbeddings 初始化完成 | model={self.model}")

    # ──────────────────────────────────────────────────────────────────────────
    # LangChain Embeddings 接口实现
    # ──────────────────────────────────────────────────────────────────────────

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入文档（自动分批，处理 API 单次限制）"""
        if not texts:
            return []

        all_embeddings: List[List[float]] = []
        batches = [
            texts[i : i + self.BATCH_SIZE]
            for i in range(0, len(texts), self.BATCH_SIZE)
        ]

        logger.debug(f"embed_documents | 共 {len(texts)} 条 | {len(batches)} 个批次")

        for batch_idx, batch in enumerate(batches):
            try:
                response = self._client.embeddings.create(
                    model=self.model,
                    input=batch,
                    encoding_format="float",
                )
                # 按原始顺序收集（API 保证顺序与 input 一致）
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                logger.debug(f"批次 {batch_idx + 1}/{len(batches)} 嵌入完成")
            except Exception as e:
                logger.error(f"批次 {batch_idx + 1} 嵌入失败：{e}")
                raise

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """单条查询嵌入"""
        logger.debug(f"embed_query | 文本长度={len(text)}")
        try:
            response = self._client.embeddings.create(
                model=self.model,
                input=[text],
                encoding_format="float",
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"查询嵌入失败：{e}")
            raise
