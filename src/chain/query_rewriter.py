"""
查询改写模块（Query Rewriting）

在检索之前，用 LLM 把用户的口语化、模糊的问题改写成更精准、
更适合检索专业文档的表达，从而提升检索命中率。

例：
    用户问："基坑挖多深算深的？"
    改写后："深基坑工程的开挖深度定义标准是多少米"

设计要点：
1. 改写失败或异常时，自动回退使用原始问题，绝不阻塞主流程
2. 改写只服务于"检索"，最终回答仍基于真实检索到的文档
3. 可通过配置开关（ENABLE_QUERY_REWRITE）控制是否启用
"""
from __future__ import annotations

from typing import Optional

from src.llm import QwenLLM
from src.utils import get_logger

logger = get_logger(__name__)


# 改写提示词：要求 LLM 只输出改写后的问题，不要多余内容
_REWRITE_PROMPT = """你是一个专业的检索查询优化助手。你的任务是把用户的原始问题改写成更适合在专业文档（如规章制度、技术规范）中检索的表达。

改写要求：
1. 保持原意不变，不要曲解用户意图
2. 把口语化表达转为规范、专业的术语
3. 补充可能有助于检索的关键词（如同义词、专业概念）
4. 如果原问题已经足够清晰专业，可以基本保持原样
5. 只输出改写后的问题，不要任何解释、前缀、引号或多余文字
6. 改写后的问题应简洁，控制在一句话内

原始问题：{question}

改写后的问题："""


class QueryRewriter:
    """查询改写器：用 LLM 优化检索查询"""

    def __init__(self, llm: Optional[QwenLLM] = None) -> None:
        # 复用同一个 LLM；改写用低温度保证稳定
        self.llm = llm or QwenLLM(temperature=0.0)
        logger.info("QueryRewriter 初始化完成")

    def rewrite(self, question: str) -> str:
        """
        改写查询。

        Args:
            question: 用户原始问题

        Returns:
            改写后的问题；若改写失败则返回原始问题（安全回退）
        """
        if not question or not question.strip():
            return question

        prompt = _REWRITE_PROMPT.format(question=question.strip())

        try:
            rewritten = self.llm._call(prompt).strip()
        except Exception as e:
            logger.warning(f"查询改写失败，回退使用原始问题：{e}")
            return question

        # 清洗：去掉模型可能多带的引号、前缀
        rewritten = self._clean(rewritten)

        # 安全校验：改写结果为空或过长（疑似跑偏）时，回退原问题
        if not rewritten or len(rewritten) > len(question) * 4 + 50:
            logger.warning("改写结果异常，回退使用原始问题")
            return question

        if rewritten != question:
            logger.info(f"查询改写：『{question}』→『{rewritten}』")
        return rewritten

    @staticmethod
    def _clean(text: str) -> str:
        """去掉模型输出里常见的多余包裹（前缀、引号），循环清理直到稳定"""
        text = text.strip()
        prefixes = ("改写后的问题：", "改写后：", "问题：", "改写结果：")
        quotes = ('"', "“", "”", "'", "「", "」")

        prev = None
        # 反复剥离，直到不再变化（处理"前缀+引号"嵌套的情况）
        while prev != text:
            prev = text
            for prefix in prefixes:
                if text.startswith(prefix):
                    text = text[len(prefix):].strip()
            for quote in quotes:
                if text.startswith(quote):
                    text = text[len(quote):].strip()
                if text.endswith(quote):
                    text = text[: -len(quote)].strip()
        return text.strip()
