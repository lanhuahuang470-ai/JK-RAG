"""
查询改写模块（Query Rewriting）

在检索之前，用 LLM 把用户的口语化、模糊的问题改写成更精准、
更适合检索专业文档的表达，从而提升检索命中率。

支持多轮对话：如果传入 chat_history，会先做上下文补全再做检索优化。

例：
    用户问："基坑挖多深算深的？"
    改写后："深基坑工程的开挖深度定义标准是多少米"
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from src.utils import get_logger

if TYPE_CHECKING:
    from src.llm import QwenLLM

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


# 带历史上下文的改写提示词：补全省略与指代，输出完整独立的问题
_HISTORY_REWRITE_PROMPT = """你是一个专业的对话上下文理解助手。给定完整的对话历史和当前用户问题，请将当前问题改写为一个**完整、独立、脱离上下文也能完全理解**的问题。

━━━ 【核心规则（必须严格遵守）】━━━
1. **必须补全省略成分**：主语、宾语、谓语、定语，凡是上文提到过且当前问题省略的，全部补全
2. **必须替换所有指代**：它、这、这个、那个、这些、那些、上面、刚才、之前、提到的、呢、的、嘛、啊 等，全部替换为具体内容
3. **"X的呢"句式**：X 是新条件，保留 X，补全上文讨论的主题和谓语。如"上海的呢"+上文讨论"南通深基坑评审要求"→"上海的深基坑评审要求有哪些"
4. **"那X呢"句式**：表示转向 X，结合上文主题补全。如"那降水呢"+上文讲"深基坑施工"→"深基坑施工中的降水方面是怎样的"
5. **"它/这个/那个"指代**：找到上文最近的名词性话题，替换进去。如"它的流程是什么"+上文讲"深基坑评审"→"深基坑评审的流程是什么"
6. **"还有/还有什么/另外"句式**：表示追加同类问题，补全主题。如"还有什么要求"+上文讲"评审"→"深基坑评审还有什么其他要求"
7. **"为什么/怎么/如何"开头的省略句**：补全主语和主题。如"为什么不行"+上文讲"某种施工方法"→"为什么这种施工方法不行"
8. **多轮对话**：以最后一轮讨论的核心话题为准，往前追溯必要的上下文
9. 如果当前问题已经是完整明确的（没有省略、没有指代），直接返回原问题
10. **只输出改写后的问题本身**，不要任何解释、前缀、引号、编号，也不要重复问题

━━━ 【示例（仔细学习）】━━━

示例1（"X的呢"句式）：
历史：
用户：南通深基坑评审要求有哪些
助手：南通深基坑评审要求包括...
当前：上海的呢
改写：上海的深基坑评审要求有哪些

示例2（"需要注意什么"省略）：
历史：
用户：基坑降水的方法有哪些
助手：基坑降水的方法主要有井点降水、明排降水等
当前：需要注意什么
改写：基坑降水施工时需要注意什么事项

示例3（"它"指代）：
历史：
用户：什么是深基坑
助手：深基坑是指开挖深度超过5米的基坑
当前：它的施工流程是什么
改写：深基坑的施工流程是什么

示例4（"那X呢"转话题）：
历史：
用户：深基坑支护有哪些形式
助手：主要有排桩、地下连续墙、水泥土墙等形式
当前：那降水方案呢
改写：深基坑的降水方案有哪些

示例5（"还有什么"追加）：
历史：
用户：深基坑评审需要提交哪些材料
助手：需要提交施工方案、专家论证表、资质证明
当前：还有什么要求
改写：深基坑评审除了材料之外还有什么其他要求

示例6（多轮对话，3轮以上）：
历史：
用户：什么是深基坑工程
助手：深基坑工程是指...
用户：深基坑支护有哪些类型
助手：主要有排桩支护、地下连续墙、土钉墙等
用户：土钉墙适用于什么地质
助手：土钉墙适用于地下水位较低、土质较好的地区
当前：那排桩呢
改写：排桩支护适用于什么地质条件

示例7（"为什么"省略句）：
历史：
用户：为什么深基坑要做专家论证
助手：因为深基坑施工风险大，需要专家把关
当前：那为什么有的不用
改写：为什么有的深基坑工程不需要做专家论证

示例8（完全问题，无需改写）：
历史：
用户：深基坑支护有哪些形式
助手：主要有排桩、地下连续墙等
当前：基坑降水的方法有哪些
改写：基坑降水的方法有哪些

━━━ 【对话历史】━━━
{history}

━━━ 【当前问题】━━━
{question}

━━━ 【改写后的完整问题】━━━"""


class QueryRewriter:
    """查询改写器：用 LLM 优化检索查询

    支持两种模式：
    - 单轮改写：只优化问题表述，提升检索命中率
    - 多轮改写：先补全上下文，再优化检索表达
    """

    def __init__(self, llm: Optional["QwenLLM"] = None) -> None:
        # 延迟导入，避免循环引用
        if llm is None:
            from src.llm import QwenLLM
            llm = QwenLLM(temperature=0.0)
        self.llm = llm
        logger.info("QueryRewriter 初始化完成")

    def rewrite(
        self,
        question: str,
        chat_history: Optional[List[dict]] = None,
    ) -> str:
        """
        改写查询。

        Args:
            question: 用户原始问题
            chat_history: 历史消息列表，用于多轮对话上下文补全

        Returns:
            改写后的问题；若改写失败则返回原始问题（安全回退）
        """
        if not question or not question.strip():
            return question

        # 有历史上下文：用带历史的改写 prompt
        if chat_history:
            prompt = self._build_history_prompt(question, chat_history)
        else:
            prompt = _REWRITE_PROMPT.format(question=question.strip())

        try:
            rewritten = self.llm._call(prompt).strip()
        except Exception as e:
            logger.warning(f"查询改写失败，回退使用原始问题：{e}")
            return question

        # 清洗：去掉模型可能多带的引号、前缀
        rewritten = self._clean(rewritten)

        # 多轮质量校验：明显是省略句但改写后几乎没变长 → 补全可能失败，重试一次
        if (
            chat_history
            and self._is_omission_question(question)
            and len(rewritten) <= len(question) + 3
        ):
            logger.warning(f"改写质量可能不足（{question} → {rewritten}），重试一次")
            recent = chat_history[-6:]
            retry_history = "\n".join(
                f"{'用户' if m.get('role') == 'user' else '助手'}：{m.get('content', '')[:300]}"
                for m in recent
            )
            retry_prompt = (
                "请仔细分析对话历史，用户当前的问题是一个省略句，"
                "你必须结合上文补全所有省略的成分，让问题完整独立。\n\n"
                f"对话历史：\n{retry_history}\n\n"
                f"当前问题：{question}\n\n"
                "请直接输出补全后的完整问题："
            )
            try:
                rewritten2 = self._clean(self.llm._call(retry_prompt).strip())
                if len(rewritten2) > len(question) + 3:
                    rewritten = rewritten2
            except Exception as e2:
                logger.warning(f"重试改写也失败：{e2}")

        # 安全校验：改写结果为空或过长（疑似跑偏）时，回退原问题
        if not rewritten or len(rewritten) > len(question) * 4 + 50:
            logger.warning("改写结果异常，回退使用原始问题")
            return question

        if rewritten != question:
            logger.info(f"查询改写：『{question}』→『{rewritten}』")
        return rewritten

    def _build_history_prompt(self, question: str, chat_history: List[dict]) -> str:
        """构建带历史上下文的改写 prompt"""
        # 只取最近 6 条（3 轮对话）
        recent = chat_history[-6:]
        history_lines = []
        for msg in recent:
            role = "用户" if msg.get("role") == "user" else "助手"
            content = msg.get("content", "")
            # 截断过长消息
            if len(content) > 300:
                content = content[:300] + "..."
            history_lines.append(f"{role}：{content}")
        history_text = "\n".join(history_lines)
        return _HISTORY_REWRITE_PROMPT.format(
            history=history_text, question=question.strip()
        )

    @staticmethod
    def _is_omission_question(question: str) -> bool:
        """判断问题是否明显是省略句（短且包含省略特征词）"""
        q = question.strip()
        if len(q) > 15:
            return False
        # 常见的省略句特征词
        omission_markers = ["呢", "的呢", "那", "那它", "这个", "那个", "它", "还有",
                            "怎么", "为什么", "什么", "怎么样", "如何", "嘛"]
        return any(q.endswith(m) or m in q for m in omission_markers)

    @staticmethod
    def _clean(text: str) -> str:
        """去掉模型输出里常见的多余包裹（前缀、引号），循环清理直到稳定"""
        text = text.strip()
        prefixes = (
            "改写后的问题：", "改写后：", "问题：", "改写结果：",
            "补全后的问题：", "完整问题：", "最终问题：",
        )
        quotes = ('"', "“", "”", "'", "「", "」", "《", "》")

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
