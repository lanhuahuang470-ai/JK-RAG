"""
RAG 问答链模块
采用 LangChain stuff 策略：将所有检索到的上下文拼接后一次性送入 LLM
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

from config.settings import settings
from src.llm import QwenLLM
from src.vectorstore import FAISSVectorStore
from src.utils import get_logger

logger = get_logger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Prompt 模板（中英双语兼容）
# ──────────────────────────────────────────────────────────────────────────────

_STUFF_PROMPT_TEMPLATE = """你是一个专业的企业知识库问答助手。
请根据以下【参考资料】回答用户的【问题】。

规则：
1. 仅根据参考资料回答，不要编造信息。
2. 如果参考资料中没有相关内容，请明确说明"抱歉，知识库中没有找到相关信息"。
3. 回答要简洁、准确、条理清晰。
4. 如有必要，可以标注信息来源（文件名）。

【参考资料】
{context}

【问题】
{question}

【回答】"""

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
        llm: Optional[QwenLLM] = None,
        top_k: int = settings.retrieval_top_k,
        enable_query_rewrite: Optional[bool] = None,
    ) -> None:
        self.vector_store = vector_store or FAISSVectorStore()
        self.llm = llm or QwenLLM()
        self.top_k = top_k
        self._chain: Optional[RetrievalQA] = None

        # 查询改写开关：参数优先，否则读配置
        self.enable_query_rewrite = (
            enable_query_rewrite
            if enable_query_rewrite is not None
            else settings.enable_query_rewrite
        )
        self._rewriter = None  # 延迟初始化（启用时才建）

        logger.info(
            f"RAGChain 初始化 | top_k={top_k} | "
            f"查询改写={'开' if self.enable_query_rewrite else '关'}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 链构建
    # ──────────────────────────────────────────────────────────────────────────

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

    # ──────────────────────────────────────────────────────────────────────────
    # 问答接口
    # ──────────────────────────────────────────────────────────────────────────

    def query(self, question: str) -> Dict[str, Any]:
        """
        执行 RAG 问答

        Args:
            question: 用户问题

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

        # ── 查询改写：检索前优化问题（启用时）──────────────────────────────
        search_query = question  # 用于检索的问题（可能被改写）
        rewritten_query = None
        if self.enable_query_rewrite:
            if self._rewriter is None:
                from src.chain.query_rewriter import QueryRewriter
                self._rewriter = QueryRewriter(llm=self.llm)
            rewritten_query = self._rewriter.rewrite(question)
            search_query = rewritten_query

        try:
            # 用（可能改写后的）问题检索并生成回答
            result = self._chain.invoke({"query": search_query})
        except Exception as e:
            logger.error(f"RAG 链执行失败：{e}")
            raise

        answer: str = result.get("result", "")
        source_docs: List[Document] = result.get("source_documents", [])

        # 额外做一次带分数的检索，拿到每个来源片段的相似度（检索置信度）
        # 注意：用与上面一致的 search_query，保证分数与实际检索匹配
        scored = self.vector_store.similarity_search_with_scores(
            search_query, k=self.top_k
        )
        # 汇总每个来源文件的最高相似度分数
        source_scores: Dict[str, float] = {}
        for doc, score in scored:
            src = doc.metadata.get("source", "未知来源")
            # 同一文件可能有多个片段，取最高分代表该来源的相关度
            if src not in source_scores or score > source_scores[src]:
                source_scores[src] = round(float(score), 3)

        # 提取唯一来源文件名
        sources = list(
            dict.fromkeys(
                doc.metadata.get("source", "未知来源")
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

        return {
            "question": question,
            "rewritten_query": rewritten_query,        # 改写后的问题（未启用则为 None）
            "answer": answer,
            "source_documents": source_docs,
            "sources": sources,
            "source_scores": source_scores,           # 每个来源的相似度
            "retrieval_confidence": round(float(top_score), 3),  # 整体检索置信度
        }

    def query_simple(self, question: str) -> str:
        """仅返回回答文本（简化接口）"""
        return self.query(question)["answer"]
