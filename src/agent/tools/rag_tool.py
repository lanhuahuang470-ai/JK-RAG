"""
RAG 工具

把现有的 RAGChain 包装成 agent 可调用的"工具"。
工具层职责：对外提供统一接口，对内调用基础层（RAGChain），
agent 大脑只跟工具打交道，不直接碰 RAGChain。
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from src.chain import RAGChain
from src.utils import get_logger

logger = get_logger(__name__)


class RAGTool:
    """本地知识库检索工具（包装 RAGChain）"""

    name = "knowledge_search"
    description = "在本地基坑工程规范知识库中检索并回答问题"

    def __init__(self, rag_chain: Optional[RAGChain] = None) -> None:
        self.chain = rag_chain or RAGChain()
        logger.info("RAGTool 初始化完成")

    def search(self, question: str) -> Dict[str, Any]:
        """
        调用本地知识库回答问题。

        Returns:
            {
                "answer": str,           # RAG 回答
                "confidence": float,     # 检索置信度（0~1）
                "sources": [str],        # 来源文件
                "source_type": "知识库",
            }
        """
        result = self.chain.query(question)
        return {
            "answer": result["answer"],
            "confidence": result["retrieval_confidence"],
            "sources": result["sources"],
            "source_scores": result.get("source_scores", {}),
            "source_type": "知识库",
        }
