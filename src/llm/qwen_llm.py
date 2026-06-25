"""
Qwen 大语言模型封装
通过 DashScope OpenAI 兼容接口调用 qwen3-7b-max
"""
from __future__ import annotations

import os
from typing import Any, Dict, Iterator, List, Optional

from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from openai import OpenAI

from config.settings import settings
from src.utils import get_logger

logger = get_logger(__name__)

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class QwenLLM(LLM):
    """
    Qwen3-7B-Max LangChain LLM 适配器

    通过 DashScope OpenAI 兼容接口调用，支持：
    - 同步生成（_call）
    - 流式生成（_stream）
    """

    model: str = settings.llm_model
    temperature: float = settings.llm_temperature
    max_tokens: int = settings.llm_max_tokens
    api_key: str = ""
    base_url: str = DASHSCOPE_BASE_URL

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # api_key 优先级：构造参数 > 环境变量 > settings
        if not self.api_key:
            object.__setattr__(
                self,
                "api_key",
                settings.dashscope_api_key or os.environ.get("DASHSCOPE_API_KEY", ""),
            )
        if not self.api_key:
            raise ValueError("DASHSCOPE_API_KEY 未配置，无法初始化 LLM。")
        logger.info(f"QwenLLM 初始化完成 | model={self.model} | temperature={self.temperature}")

    @property
    def _llm_type(self) -> str:
        return "qwen-dashscope"

    @property
    def _client(self) -> OpenAI:
        return OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> str:
        """同步推理（RetrievalQA 链默认使用此方法）"""
        logger.debug(f"LLM 推理 | prompt_len={len(prompt)}")
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stop=stop,
                **kwargs,
            )
            answer = response.choices[0].message.content or ""
            logger.debug(f"LLM 响应 | answer_len={len(answer)}")
            return answer
        except Exception as e:
            logger.error(f"LLM 推理失败：{e}")
            raise

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """流式推理"""
        logger.debug(f"LLM 流式推理 | prompt_len={len(prompt)}")
        try:
            stream = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stop=stop,
                stream=True,
                **kwargs,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    yield GenerationChunk(text=delta)
        except Exception as e:
            logger.error(f"LLM 流式推理失败：{e}")
            raise

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
