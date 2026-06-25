"""
RAG + 网络兜底 Agent

策略（固定流程，编排式）：
1. 先查本地知识库（RAG），拿到答案 + 检索置信度
2. 按"阈值 + LLM 临界判断"决定是否够好：
   - 置信度 ≥ 高阈值（默认 0.65）→ 直接用 RAG
   - 置信度 ≤ 低阈值（默认 0.45）→ 直接转网络
   - 介于两者之间（临界区）→ 让 LLM 判断 RAG 答案是否靠谱
3. 需要兜底时调用网络搜索

这是"大脑"层：只负责决策和编排，通过工具调用基础能力，
不直接操作 RAGChain / Tavily。
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from config.settings import settings
from src.agent.tools.rag_tool import RAGTool
from src.llm import QwenLLM
from src.utils import get_logger

logger = get_logger(__name__)


# LLM 临界判断提示词：判断 RAG 答案是否足够回答问题
_JUDGE_PROMPT = """你是一个回答质量评估助手。请判断下面这个"基于本地知识库的回答"是否充分、可靠地回答了用户问题。

判断标准：
- 如果回答确实解决了问题、信息明确具体 → 输出：可靠
- 如果回答含糊、说"没有找到""未提及""无相关信息"、或明显答非所问 → 输出：不可靠

只输出"可靠"或"不可靠"两个字之一，不要任何解释。

用户问题：{question}

本地知识库的回答：{answer}

判断："""


class FallbackAgent:
    """RAG 优先、网络兜底的问答 Agent"""

    def __init__(
        self,
        rag_tool: Optional[RAGTool] = None,
        llm: Optional[QwenLLM] = None,
        enable_web_fallback: Optional[bool] = None,
        high_threshold: float = settings.fallback_high_threshold,
        low_threshold: float = settings.fallback_low_threshold,
    ) -> None:
        self.rag_tool = rag_tool or RAGTool()
        self.llm = llm or QwenLLM()
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.enable_web_fallback = (
            enable_web_fallback
            if enable_web_fallback is not None
            else settings.enable_web_fallback
        )
        self._web_tool = None  # 延迟初始化（需要时才建，避免无 key 时报错）

        logger.info(
            f"FallbackAgent 初始化 | 网络兜底={'开' if self.enable_web_fallback else '关'} | "
            f"高阈值={high_threshold} | 低阈值={low_threshold}"
        )

    # ──────────────────────────────────────────────────────────────────────────

    def query(self, question: str) -> Dict[str, Any]:
        """
        执行问答（RAG 优先，必要时网络兜底）。

        Returns:
            {
                "answer": str,
                "source_type": "知识库" | "网络",
                "confidence": float,        # RAG 置信度
                "sources": [str],
                "route": str,               # 决策路径说明
            }
        """
        # 步骤1：先查 RAG
        rag_result = self.rag_tool.search(question)
        conf = rag_result["confidence"]
        logger.info(f"RAG 检索置信度={conf}")

        # 网络兜底未启用：直接返回 RAG 结果
        if not self.enable_web_fallback:
            rag_result["route"] = "仅 RAG（网络兜底未启用）"
            return rag_result

        # 步骤2：决策
        if conf >= self.high_threshold:
            # 高置信度：直接用 RAG
            rag_result["route"] = f"RAG（置信度 {conf} ≥ {self.high_threshold}）"
            return rag_result

        if conf <= self.low_threshold:
            # 低置信度：直接转网络
            logger.info(f"置信度过低（{conf} ≤ {self.low_threshold}），转网络搜索")
            return self._do_web_search(question, conf, reason=f"置信度低（{conf}）")

        # 临界区：让 LLM 判断 RAG 答案是否靠谱
        logger.info(
            f"置信度临界（{self.low_threshold} < {conf} < {self.high_threshold}），LLM 判断"
        )
        if self._judge_rag_reliable(question, rag_result["answer"]):
            rag_result["route"] = "RAG（临界区，LLM 判定可靠）"
            return rag_result
        else:
            logger.info("LLM 判定 RAG 答案不可靠，转网络搜索")
            return self._do_web_search(question, conf, reason="临界区 LLM 判定不可靠")

    # ──────────────────────────────────────────────────────────────────────────

    def _judge_rag_reliable(self, question: str, answer: str) -> bool:
        """让 LLM 判断 RAG 答案是否可靠"""
        prompt = _JUDGE_PROMPT.format(question=question, answer=answer)
        try:
            verdict = self.llm._call(prompt).strip()
        except Exception as e:
            logger.warning(f"LLM 判断失败，保守地认为 RAG 可靠：{e}")
            return True  # 判断失败时保守用 RAG，不浪费网络调用
        reliable = "可靠" in verdict and "不可靠" not in verdict
        logger.info(f"LLM 判定：{verdict}")
        return reliable

    def _do_web_search(
        self, question: str, rag_conf: float, reason: str
    ) -> Dict[str, Any]:
        """执行网络搜索兜底"""
        if self._web_tool is None:
            from src.agent.tools.web_search_tool import WebSearchTool
            self._web_tool = WebSearchTool(llm=self.llm)

        web_result = self._web_tool.search(question)
        web_result["confidence"] = rag_conf  # 附带原 RAG 置信度供参考
        web_result["route"] = f"网络兜底（{reason}）"
        return web_result
