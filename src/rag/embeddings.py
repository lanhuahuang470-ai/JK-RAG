"""
Qwen3-Embedding 嵌入模型封装
通过 DashScope OpenAI 兼容接口调用 text-embedding-v4（Qwen3-Embedding）

性能优化：
- 客户端复用（全局缓存，避免每次新建 HTTP 连接）
- 查询嵌入缓存（相同问题直接返回缓存结果，节省 API 调用）
- 连接池配置（timeout + max_retries）
- 批量嵌入自动分批
"""
from __future__ import annotations

import hashlib
import os
from typing import Dict, List

from langchain_core.embeddings import Embeddings
from openai import OpenAI

from config.settings import settings
from src.utils import get_logger

logger = get_logger(__name__)

# DashScope OpenAI 兼容端点
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 全局客户端缓存
_embedding_clients: Dict[str, OpenAI] = {}

# 查询嵌入缓存（LRU 由 dict 自然顺序 + 容量控制）
_embedding_cache: Dict[str, List[float]] = {}
_CACHE_MAX_SIZE = 1000  # 最多缓存 1000 条查询嵌入


def _get_cached_client(api_key: str) -> OpenAI:
    """获取或创建缓存的 OpenAI 客户端"""
    if api_key not in _embedding_clients:
        _embedding_clients[api_key] = OpenAI(
            api_key=api_key,
            base_url=DASHSCOPE_BASE_URL,
            timeout=settings.embedding_timeout,
            max_retries=settings.embedding_max_retries,
        )
        logger.debug("新建 Embedding 客户端")
    return _embedding_clients[api_key]


def _cache_get(text: str) -> List[float] | None:
    """从缓存获取嵌入向量"""
    if not settings.enable_embedding_cache:
        return None
    key = hashlib.md5(text.encode("utf-8")).hexdigest()
    result = _embedding_cache.get(key)
    if result is not None:
        # 命中：移到末尾（模拟 LRU，最近使用的在后面）
        del _embedding_cache[key]
        _embedding_cache[key] = result
    return result


def _cache_set(text: str, embedding: List[float]) -> None:
    """写入缓存，超出容量时淘汰最旧的"""
    if not settings.enable_embedding_cache:
        return
    key = hashlib.md5(text.encode("utf-8")).hexdigest()
    if key in _embedding_cache:
        del _embedding_cache[key]
    _embedding_cache[key] = embedding
    # 超出容量时，淘汰最前面的（最久未使用）
    while len(_embedding_cache) > _CACHE_MAX_SIZE:
        _embedding_cache.pop(next(iter(_embedding_cache)))


class DashScopeEmbeddings(Embeddings):
    """
    Qwen3-Embedding（text-embedding-v4）LangChain 适配器

    通过 DashScope 的 OpenAI 兼容接口调用，支持：
    - 批量嵌入（embed_documents）
    - 单条查询嵌入（embed_query）
    - 查询嵌入缓存（相同查询直接返回缓存）

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

        self._api_key = _key
        self._client = _get_cached_client(_key)
        logger.info(
            f"DashScopeEmbeddings 初始化完成 | model={self.model} | "
            f"缓存={'开' if settings.enable_embedding_cache else '关'}"
        )

    # ── LangChain Embeddings 接口实现 ──────────────────────────────────────

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
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                logger.debug(f"批次 {batch_idx + 1}/{len(batches)} 嵌入完成")
            except Exception as e:
                logger.error(f"批次 {batch_idx + 1} 嵌入失败：{e}")
                raise

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """单条查询嵌入（带缓存）"""
        # 先查缓存
        cached = _cache_get(text)
        if cached is not None:
            logger.debug(f"embed_query 缓存命中 | 文本长度={len(text)}")
            return cached

        logger.debug(f"embed_query | 文本长度={len(text)}")
        try:
            response = self._client.embeddings.create(
                model=self.model,
                input=[text],
                encoding_format="float",
            )
            embedding = response.data[0].embedding
            # 写入缓存
            _cache_set(text, embedding)
            return embedding
        except Exception as e:
            logger.error(f"查询嵌入失败：{e}")
            raise
