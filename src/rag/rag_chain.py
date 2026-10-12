"""
RAG 问答链模块
采用 LangChain stuff 策略：将所有检索到的上下文拼接后一次性送入 LLM

性能优化：
- 检索结果缓存（相同查询直接返回缓存的检索结果）
- LLM 客户端复用（由 QwenLLM 提供）
- 嵌入缓存（由 DashScopeEmbeddings 提供）
"""
from __future__ import annotations

import hashlib
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

from config.settings import settings
from src.rag.vectorstore import FAISSVectorStore
from src.utils import get_logger

if TYPE_CHECKING:
    from src.llm import QwenLLM

logger = get_logger(__name__)

# 检索结果缓存（LRU）
_retrieval_cache: Dict[str, List[Tuple[Document, float]]] = {}
_RETRIEVAL_CACHE_MAX = 200  # 最多缓存 200 条检索结果


def _retrieval_cache_key(query: str, k: int) -> str:
    """生成检索缓存的 key"""
    return hashlib.md5(f"{k}::{query}".encode("utf-8")).hexdigest()


def _retrieval_cache_get(query: str, k: int) -> Optional[List[Tuple[Document, float]]]:
    """从缓存获取检索结果"""
    if not settings.enable_retrieval_cache:
        return None
    key = _retrieval_cache_key(query, k)
    result = _retrieval_cache.get(key)
    if result is not None:
        # LRU：命中后移到末尾
        del _retrieval_cache[key]
        _retrieval_cache[key] = result
    return result


def _retrieval_cache_set(query: str, k: int, result: List[Tuple[Document, float]]) -> None:
    """写入检索缓存"""
    if not settings.enable_retrieval_cache:
        return
    key = _retrieval_cache_key(query, k)
    if key in _retrieval_cache:
        del _retrieval_cache[key]
    _retrieval_cache[key] = result
    # 超出容量时淘汰最旧的
    while len(_retrieval_cache) > _RETRIEVAL_CACHE_MAX:
        _retrieval_cache.pop(next(iter(_retrieval_cache)))


def clear_retrieval_cache() -> None:
    """清空检索缓存（索引更新后调用）"""
    _retrieval_cache.clear()
    logger.info("检索缓存已清空")

# ── Prompt 模板（中英双语兼容）─────────────────────────────────────────

_STUFF_PROMPT_TEMPLATE = """你是一个专业的企业知识库问答助手，正在与用户进行多轮对话。

{history_section}
请根据以下【参考资料】回答用户的【当前问题】。

规则：
1. 仅根据参考资料回答，不要编造信息。
2. 如果参考资料中没有相关内容，请明确说明"抱歉，知识库中没有找到相关信息"。
3. 回答要简洁、准确、条理清晰。
4. 如有必要，可以标注信息来源（文件名）。
5. 请结合对话历史理解用户问题的真实含义，但回答内容必须以参考资料为准。

【参考资料】
{context}

【当前问题】
{question}

【回答】"""

_HISTORY_PREFIX = """【对话历史】
{history}
"""

STUFF_PROMPT = PromptTemplate(
    template=_STUFF_PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)


class RAGChain:
    """
    企业级 RAG 问答链

    架构：
    用户问题
        ↓
    FAISS 相似度检索 → Top-K 文档
        ↓
    Stuff 策略：拼接所有上下文
        ↓
    QwenLLM 生成回答
        ↓
    结构化输出（含来源）
    """

    def __init__(
        self,
        vector_store: Optional[FAISSVectorStore] = None,
        llm: Optional["QwenLLM"] = None,
        top_k: int = settings.retrieval_top_k,
        enable_query_rewrite: Optional[bool] = None,
        enable_hybrid: Optional[bool] = None,
        enable_rerank: Optional[bool] = None,
    ) -> None:
        self.vector_store = vector_store or FAISSVectorStore()
        # 延迟导入 LLM，避免循环引用
        if llm is None:
            from src.llm import QwenLLM
            llm = QwenLLM()
        self.llm = llm
        self.top_k = top_k
        self._chain: Optional[RetrievalQA] = None

        # 查询改写开关：参数优先，否则读配置
        self.enable_query_rewrite = (
            enable_query_rewrite
            if enable_query_rewrite is not None
            else settings.enable_query_rewrite
        )
        self._rewriter = None  # 延迟初始化（启用时才建）

        # 混合检索 / Rerank 开关：参数优先，否则读配置
        self.enable_hybrid = (
            enable_hybrid
            if enable_hybrid is not None
            else settings.enable_hybrid_retrieval
        )
        self.enable_rerank = (
            enable_rerank if enable_rerank is not None else settings.enable_rerank
        )
        self._hybrid = None    # 延迟初始化（首次检索时建 BM25 索引）
        self._reranker = None  # 延迟初始化

        logger.info(
            f"RAGChain 初始化 | top_k={top_k} | "
            f"查询改写={'开' if self.enable_query_rewrite else '关'} | "
            f"混合检索={'开' if self.enable_hybrid else '关'} | "
            f"Rerank={'开' if self.enable_rerank else '关'}"
        )

    # ── 链构建 ────────────────────────────────────────────────────────────

    def _build_chain(self) -> RetrievalQA:
        """
        构建 RetrievalQA（stuff 策略）链

        stuff 策略说明：
        - chain_type="stuff"：将所有检索文档拼接为单一上下文一次送入 LLM
        - 适合 Top-K 较小（≤5）、单文档上下文窗口充足的场景
        - 优点：延迟低、实现简单；缺点：上下文过长时可能超出模型 token 限制
        """
        retriever = self.vector_store.get_retriever(k=self.top_k)

        chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",                      # ← 核心：stuff 策略
            retriever=retriever,
            return_source_documents=True,            # 返回来源文档
            chain_type_kwargs={
                "prompt": STUFF_PROMPT,              # 自定义中文 Prompt
                "verbose": False,
            },
        )
        logger.info("RetrievalQA (stuff) 链构建完成")
        return chain

    def _ensure_chain(self) -> None:
        """延迟初始化：首次调用时构建链"""
        if self._chain is None:
            self._chain = self._build_chain()

    # ── 三段式检索：召回 → 融合 → 精排 ─────────────────────────────────────

    def _get_hybrid(self):
        """延迟初始化混合检索器（首次检索时构建 BM25 索引）"""
        if self._hybrid is None and self.enable_hybrid:
            from src.rag.hybrid_retriever import HybridRetriever
            self._hybrid = HybridRetriever(self.vector_store)
        return self._hybrid

    def _get_reranker(self):
        """延迟初始化 Rerank 精排器"""
        if self._reranker is None and self.enable_rerank:
            from src.rag.reranker import DashScopeReranker
            self._reranker = DashScopeReranker()
        return self._reranker

    @staticmethod
    def _doc_key(doc: Document) -> str:
        return str(hash(doc.page_content))

    def _retrieve(
        self, query: str
    ) -> Tuple[List[Tuple[Document, float]], Dict[str, float]]:
        """
        三段式检索管线：
        1. 召回：向量 + BM25 混合（未启用则为纯向量），取 fetch_k 个候选
        2. RRF 融合（混合模式在 HybridRetriever 内部完成）
        3. Rerank 精排：cross-encoder 对候选重排，截断到 top_k

        Returns:
            scored: [(Document, 分数)]。启用 Rerank 时为 relevance_score，
                    否则为向量相关性分数，均保持 0~1 置信度语义
            stage_timings: 各阶段耗时明细（recall_ms / rerank_ms）
        """
        stage_timings: Dict[str, float] = {}
        fetch_k = max(settings.retrieval_fetch_k, self.top_k)

        # ── 第一段：召回（混合 or 纯向量）──
        recall_t0 = time.perf_counter()
        hybrid = self._get_hybrid()
        if hybrid is not None:
            candidates, vector_scores = hybrid.retrieve(query, fetch_k)
        else:
            hits = self.vector_store.similarity_search_with_scores(
                query, k=fetch_k
            )
            candidates = [doc for doc, _ in hits]
            vector_scores = {self._doc_key(doc): float(s) for doc, s in hits}
        stage_timings["recall_ms"] = round(
            (time.perf_counter() - recall_t0) * 1000, 2
        )

        # ── 第二段：Rerank 精排（候选多于 top_k 才有精排价值）──
        reranker = self._get_reranker()
        if reranker is not None and len(candidates) > self.top_k:
            rerank_t0 = time.perf_counter()
            scored = reranker.rerank(query, candidates, top_n=self.top_k)
            stage_timings["rerank_ms"] = round(
                (time.perf_counter() - rerank_t0) * 1000, 2
            )
        else:
            scored = [
                (doc, vector_scores.get(self._doc_key(doc), 0.0))
                for doc in candidates[: self.top_k]
            ]
        return scored, stage_timings

    # ── 问答接口 ──────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        chat_history: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        执行 RAG 问答

        Args:
            question: 用户问题
            chat_history: 历史消息列表，格式如 [{"role": "user", "content": "..."}, ...]

        Returns:
            {
                "answer": str,             # LLM 生成的回答
                "source_documents": [...], # 检索到的原始文档
                "sources": [str],          # 来源文件名列表（去重）
                "question": str,           # 原始问题
            }
        """
        self._ensure_chain()
        logger.info(f"收到问题：{question[:80]}...")
        timings_ms: Dict[str, float] = {}

        # ── 查询改写：只在多轮对话时启用（单轮不需要，省一轮 LLM 调用约 4 秒）
        search_query = question  # 用于检索的问题（可能被改写）
        rewritten_query = None
        if chat_history and self.enable_query_rewrite:
            if self._rewriter is None:
                from src.rag.query_rewriter import QueryRewriter
                self._rewriter = QueryRewriter(llm=self.llm)
            rewrite_t0 = time.perf_counter()
            rewritten_query = self._rewriter.rewrite(question, chat_history=chat_history)
            timings_ms["query_rewrite_ms"] = round(
                (time.perf_counter() - rewrite_t0) * 1000, 2
            )
            search_query = rewritten_query

        try:
            retrieval_t0 = time.perf_counter()
            # 先查缓存
            cached = _retrieval_cache_get(search_query, self.top_k)
            if cached is not None:
                scored = cached
                timings_ms["retrieval_ms"] = round(
                    (time.perf_counter() - retrieval_t0) * 1000, 2
                )
                timings_ms["retrieval_cache_hit"] = True
                logger.debug(f"检索缓存命中 | query={search_query[:40]}...")
            else:
                retrieval_t0 = time.perf_counter()
                scored, stage_timings = self._retrieve(search_query)
                timings_ms["retrieval_ms"] = round(
                    (time.perf_counter() - retrieval_t0) * 1000, 2
                )
                timings_ms.update(stage_timings)  # recall_ms / rerank_ms 明细
                _retrieval_cache_set(search_query, self.top_k, scored)
                timings_ms["retrieval_cache_hit"] = False
        except Exception as e:
            logger.error(f"RAG 链执行失败：{e}")
            raise

        source_docs: List[Document] = [doc for doc, _ in scored]

        # 构建带历史上下文的 prompt（如果有历史）
        history_text = ""
        if chat_history:
            # 只取最近 6 条（3 轮对话），兼顾上下文和 token 效率
            recent = chat_history[-6:]
            lines = []
            for msg in recent:
                role = "用户" if msg.get("role") == "user" else "助手"
                content = msg.get("content", "")
                # 截断过长的回答，避免淹没问题和检索结果
                if len(content) > 400:
                    content = content[:400] + "..."
                lines.append(f"{role}：{content}")
            history_text = _HISTORY_PREFIX.format(history="\n".join(lines))

        llm_t0 = time.perf_counter()
        try:
            # 动态构建 prompt（根据是否有历史）
            prompt_text = _STUFF_PROMPT_TEMPLATE.format(
                context="{context}",
                question=search_query,
                history_section=history_text,
            )
            # 注意：context 是 LangChain 内部替换的，这里保留占位符
            # 我们手动拼出最终的 prompt 文本
            context_text = "\n\n".join(
                f"--- 来源：{doc.metadata.get('source', '未知')} ---\n{doc.page_content}"
                for doc in source_docs
            )
            final_prompt = prompt_text.replace("{context}", context_text)
            answer = self.llm._call(final_prompt)
        except Exception as e:
            logger.error(f"RAG LLM 生成失败：{e}")
            raise
        timings_ms["llm_ms"] = round((time.perf_counter() - llm_t0) * 1000, 2)

        # 汇总每个来源文件的最高相似度分数
        source_scores: Dict[str, float] = {}
        for doc, score in scored:
            src = doc.metadata.get("source", "未知来源")
            # 同一文件可能有多个片段，取最高分代表该来源的相关度
            if src not in source_scores or score > source_scores[src]:
                source_scores[src] = round(float(score), 3)

        # 条款级来源：文件名 + 层次路径（如 "细则.pdf · 第三章 > 第十二条"），
        # 让层次切分的元数据价值透传到最终展示
        # 条款级来源：文件名 + 规则ID + 层次路径（如 "细则.pdf [1-3] · 第一章 > 第三条"）
        # rule_id 优先从 metadata 读（新索引），回退到从 hierarchy_path 实时解析（兼容旧索引）
        import re as _re

        def _cn_to_int(s: str) -> int:
            """中文数字转 int：'三'→3, '十四'→14, '二十三'→23"""
            digit = {"一":1,"二":2,"三":3,"四":4,"五":5,
                     "六":6,"七":7,"八":8,"九":9}
            if s == "十": return 10
            if s.startswith("十"): return 10 + digit.get(s[1], 0)
            if s.endswith("十"): return digit.get(s[0], 0) * 10
            if "十" in s:
                parts = s.split("十")
                return digit.get(parts[0], 0) * 10 + digit.get(parts[1], 0)
            return digit.get(s, 0)

        def _parse_rule_id(doc: Document) -> str:
            rid = doc.metadata.get("rule_id", "")
            if rid:
                return rid
            path = doc.metadata.get("hierarchy_path", "")
            if not path:
                return ""
            ch_m = _re.search(r"第([一二三四五六七八九十]+)章", path)
            art_m = _re.search(r"第([一二三四五六七八九十]+)条", path)
            ch = str(_cn_to_int(ch_m.group(1))) if ch_m else ""
            art = str(_cn_to_int(art_m.group(1))) if art_m else ""
            return f"{ch}-{art}" if ch and art else ""

        sources = list(
            dict.fromkeys(
                doc.metadata.get("source", "未知来源")
                + (
                    f" [{_parse_rule_id(doc)}]"
                    if _parse_rule_id(doc)
                    else ""
                )
                + (
                    f" · {doc.metadata['hierarchy_path']}"
                    if doc.metadata.get("hierarchy_path")
                    else ""
                )
                for doc in source_docs
            )
        )

        # 整体检索置信度 = 所有命中片段里的最高分
        top_score = max((s for _, s in scored), default=0.0)

        logger.info(
            f"问答完成 | answer_len={len(answer)} | "
            f"源文档数={len(source_docs)} | 来源={sources} | "
            f"最高相似度={round(float(top_score), 3)}"
        )
        timings_ms["rag_ms"] = round(sum(timings_ms.values()), 2)

        return {
            "question": question,
            "rewritten_query": rewritten_query,        # 改写后的问题（未启用则为 None）
            "answer": answer,
            "source_documents": source_docs,
            "sources": sources,
            "source_scores": source_scores,           # 每个来源的相似度
            "retrieval_confidence": round(float(top_score), 3),  # 整体检索置信度
            "timings_ms": timings_ms,
        }

    def query_simple(self, question: str) -> str:
        """仅返回回答文本（简化接口）"""
        return self.query(question)["answer"]
