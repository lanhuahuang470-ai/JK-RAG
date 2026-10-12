"""
混合检索器：向量召回 + BM25 关键词召回，RRF 融合排序

为什么需要混合检索：
- 向量检索擅长语义相似，但对条款编号（"第三条"）、专有名词等精确匹配偏弱
- BM25 基于词频精确匹配，与向量互补
- RRF（Reciprocal Rank Fusion）只用排名做融合，不需要对两路分数做归一化，工程上最稳
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from langchain_core.documents import Document

from src.rag.vectorstore import FAISSVectorStore
from src.utils import get_logger

logger = get_logger(__name__)

# RRF 平滑常数，业界通用经验值
_RRF_K = 60


def _tokenize(text: str) -> List[str]:
    """中文分词：BM25 默认按空格切分对中文完全无效，必须用 jieba"""
    import jieba

    return [t for t in jieba.lcut(text) if t.strip()]


class HybridRetriever:
    """
    向量 + BM25 混合召回器

    BM25 语料直接复用 FAISS 的 docstore（同一份 chunk），
    索引随向量库变化惰性重建，无需额外的持久化维护。
    """

    def __init__(self, vector_store: FAISSVectorStore) -> None:
        self.vector_store = vector_store
        self._bm25 = None
        self._bm25_doc_count: int = -1  # 记录语料规模，索引变化时触发重建

    def _ensure_bm25(self, fetch_k: int) -> bool:
        """确保 BM25 索引可用（惰性构建/重建）；失败时回退纯向量"""
        store = self.vector_store._store
        if store is None:
            return False
        total = store.index.ntotal
        if self._bm25 is not None and self._bm25_doc_count == total:
            return True
        try:
            from langchain_community.retrievers import BM25Retriever

            # FAISS docstore 保存着全部 chunk 文档，直接作为 BM25 语料
            docs = list(store.docstore._dict.values())
            if not docs:
                return False
            self._bm25 = BM25Retriever.from_documents(
                docs, preprocess_func=_tokenize
            )
            self._bm25.k = fetch_k
            self._bm25_doc_count = total
            logger.info(f"BM25 索引构建完成 | 语料 chunk 数={len(docs)}")
            return True
        except Exception as e:
            logger.warning(f"BM25 构建失败，回退纯向量检索：{e}")
            self._bm25 = None
            return False

    @staticmethod
    def _doc_key(doc: Document) -> str:
        # 向量与 BM25 返回的是不同 Document 实例，用内容哈希做去重键
        return str(hash(doc.page_content))

    def retrieve(
        self, query: str, fetch_k: int
    ) -> Tuple[List[Document], Dict[str, float]]:
        """
        混合召回：两路各取 fetch_k，RRF 融合后返回 fetch_k 个候选

        Returns:
            candidates: RRF 融合排序后的候选文档
            vector_scores: 内容哈希 → 向量相关性分数（供置信度展示，保持 0~1 语义）
        """
        # 向量一路（必走）
        vector_hits = self.vector_store.similarity_search_with_scores(
            query, k=fetch_k
        )
        vector_scores = {
            self._doc_key(doc): float(score) for doc, score in vector_hits
        }

        # BM25 一路（不可用时退化为纯向量）
        if not self._ensure_bm25(fetch_k):
            return [doc for doc, _ in vector_hits], vector_scores
        bm25_docs = self._bm25.invoke(query)

        # RRF 融合：score = Σ 1/(K + rank)
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}
        for rank, (doc, _) in enumerate(vector_hits):
            key = self._doc_key(doc)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (_RRF_K + rank + 1)
            doc_map[key] = doc
        for rank, doc in enumerate(bm25_docs):
            key = self._doc_key(doc)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (_RRF_K + rank + 1)
            doc_map.setdefault(key, doc)

        ranked_keys = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
        candidates = [doc_map[key] for key in ranked_keys[:fetch_k]]

        logger.debug(
            f"混合召回 | 向量={len(vector_hits)} | BM25={len(bm25_docs)} | "
            f"融合候选={len(candidates)}"
        )
        return candidates, vector_scores
