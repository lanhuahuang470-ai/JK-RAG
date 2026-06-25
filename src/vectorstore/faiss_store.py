"""
FAISS 向量库管理模块
封装索引的创建、持久化、加载、增量更新和相似度检索
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Optional, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config.settings import settings
from src.embeddings import DashScopeEmbeddings
from src.utils import get_logger

logger = get_logger(__name__)


class FAISSVectorStore:
    """
    企业级 FAISS 向量库管理器

    功能：
    - 从 Document 列表构建/增量更新索引
    - 持久化到磁盘 & 热加载
    - 相似度检索（含得分过滤）
    - 索引统计信息
    """

    def __init__(
        self,
        embeddings: Optional[DashScopeEmbeddings] = None,
        index_path: str | Path = settings.faiss_index_path,
    ) -> None:
        self.embeddings = embeddings or DashScopeEmbeddings()
        self.index_path = Path(index_path)
        self._store: Optional[FAISS] = None

        logger.info(f"FAISSVectorStore 初始化 | index_path={self.index_path}")

    # ──────────────────────────────────────────────────────────────────────────
    # 索引构建 & 更新
    # ──────────────────────────────────────────────────────────────────────────

    def build_index(self, documents: List[Document]) -> None:
        """
        从文档列表全量构建新索引（会覆盖已有索引）

        Args:
            documents: 切分后的 LangChain Document 列表
        """
        if not documents:
            raise ValueError("文档列表为空，无法构建索引。")

        logger.info(f"开始构建 FAISS 索引 | 文档数={len(documents)}")
        self._store = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings,
        )
        self._persist()
        logger.info("FAISS 索引构建完成并已持久化")

    def add_documents(self, documents: List[Document]) -> None:
        """
        增量添加文档到现有索引（索引不存在则自动创建）

        Args:
            documents: 新增文档列表
        """
        if not documents:
            logger.warning("增量文档列表为空，跳过。")
            return

        if self._store is None:
            self.load_index()

        if self._store is None:
            # 首次使用，直接建索引
            logger.info("索引不存在，转为全量构建模式")
            self.build_index(documents)
            return

        logger.info(f"增量添加文档 | 新增 chunk 数={len(documents)}")
        self._store.add_documents(documents)
        self._persist()
        logger.info("增量更新完成并已持久化")

    # ──────────────────────────────────────────────────────────────────────────
    # 持久化 & 加载
    # ──────────────────────────────────────────────────────────────────────────

    def _persist(self) -> None:
        """将内存中的索引持久化到磁盘"""
        if self._store is None:
            raise RuntimeError("没有可持久化的索引。")
        self.index_path.mkdir(parents=True, exist_ok=True)
        self._store.save_local(str(self.index_path))
        logger.debug(f"索引已保存至 {self.index_path}")

    def load_index(self) -> bool:
        """
        从磁盘加载索引

        Returns:
            True 表示加载成功，False 表示索引文件不存在
        """
        index_file = self.index_path / "index.faiss"
        if not index_file.exists():
            logger.warning(f"索引文件不存在：{index_file}")
            return False

        self._store = FAISS.load_local(
            folder_path=str(self.index_path),
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True,  # 受控环境下启用
        )
        logger.info(f"FAISS 索引已从磁盘加载 | path={self.index_path}")
        return True

    def is_loaded(self) -> bool:
        """索引是否已加载到内存"""
        return self._store is not None

    def reset_index(self) -> None:
        """清空索引（内存 + 磁盘）"""
        self._store = None
        if self.index_path.exists():
            shutil.rmtree(self.index_path)
            logger.warning(f"索引已清空：{self.index_path}")

    # ──────────────────────────────────────────────────────────────────────────
    # 检索
    # ──────────────────────────────────────────────────────────────────────────

    def similarity_search(
        self,
        query: str,
        k: int = settings.retrieval_top_k,
        score_threshold: float = settings.retrieval_score_threshold,
    ) -> List[Document]:
        """
        相似度检索（带得分过滤）

        Args:
            query: 用户查询文本
            k: 返回 Top-K 结果
            score_threshold: 余弦相似度阈值（0~1），低于此值的结果被过滤

        Returns:
            List[Document]：按相似度降序排列的文档列表
        """
        self._ensure_loaded()

        docs_with_scores: List[Tuple[Document, float]] = (
            self._store.similarity_search_with_relevance_scores(query, k=k)
        )

        filtered = [
            doc
            for doc, score in docs_with_scores
            if score >= score_threshold
        ]

        logger.debug(
            f"检索完成 | query_len={len(query)} | "
            f"候选={len(docs_with_scores)} | 过滤后={len(filtered)}"
        )
        return filtered

    def similarity_search_with_scores(
        self,
        query: str,
        k: int = settings.retrieval_top_k,
    ) -> List[Tuple[Document, float]]:
        """
        相似度检索，返回 (文档, 相似度分数) 列表

        分数范围 0~1，越接近 1 表示与查询越相关。
        用于向用户展示"检索置信度"。

        Args:
            query: 用户查询文本
            k: 返回 Top-K 结果

        Returns:
            List[Tuple[Document, float]]：按相似度降序排列
        """
        self._ensure_loaded()
        return self._store.similarity_search_with_relevance_scores(query, k=k)

    def get_retriever(
        self,
        k: int = settings.retrieval_top_k,
    ):
        """
        获取 LangChain 标准 Retriever（供 RetrievalQA 链使用）

        Args:
            k: Top-K 检索数量

        Returns:
            VectorStoreRetriever
        """
        self._ensure_loaded()
        return self._store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 工具方法
    # ──────────────────────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        """确保索引已加载，否则尝试从磁盘加载"""
        if self._store is None:
            loaded = self.load_index()
            if not loaded:
                raise RuntimeError(
                    "FAISS 索引尚未构建，请先调用 build_index() 或上传文档。"
                )

    @property
    def document_count(self) -> int:
        """索引中的向量数量（即 chunk 数量）"""
        if self._store is None:
            return 0
        return self._store.index.ntotal
