"""
Qwen 大语言模型封装
通过 DashScope OpenAI 兼容接口调用 qwen3-7b-max

性能优化：
- 客户端复用（单例模式，避免每次新建 HTTP 连接）
- 连接池配置（max_retries + timeout）
- 请求超时控制
- 失败自动重试（指数退避）
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, Iterator, List, Optional

from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from openai import OpenAI

from config.settings import settings
from src.utils import get_logger

logger = get_logger(__name__)

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 全局客户端缓存（按 api_key 复用）
_llm_clients: Dict[str, OpenAI] = {}


def _get_cached_client(api_key: str, base_url: str) -> OpenAI:
    """获取或创建缓存的 OpenAI 客户端（复用 HTTP 连接池）"""
    cache_key = f"{api_key}::{base_url}"
    if cache_key not in _llm_clients:
        _llm_clients[cache_key] = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=settings.llm_timeout,
            max_retries=settings.llm_max_retries,
        )
        logger.debug(f"新建 LLM 客户端 | cache_key={cache_key[:20]}...")
    return _llm_clients[cache_key]


class QwenLLM(LLM):
    """
    Qwen3-7B-Max LangChain LLM 适配器

    通过 DashScope OpenAI 兼容接口调用，支持：
    - 同步生成（_call）
    - 流式生成（_stream）
    - 自动重试（网络失败时指数退避）
    """

    model: str = settings.llm_model
    temperature: float = settings.llm_temperature
    max_tokens: int = settings.llm_max_tokens
    api_key: str = ""
    base_url: str = DASHSCOPE_BASE_URL
    _cached_client: Optional[OpenAI] = None

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
        # 预缓存客户端
        object.__setattr__(
            self,
            "_cached_client",
            _get_cached_client(self.api_key, self.base_url),
        )
        logger.info(f"QwenLLM 初始化完成 | model={self.model} | temperature={self.temperature}")

    @property
    def _llm_type(self) -> str:
        return "qwen-dashscope"

    @property
    def _client(self) -> OpenAI:
        """获取复用的客户端实例"""
        if self._cached_client is None:
            object.__setattr__(
                self,
                "_cached_client",
                _get_cached_client(self.api_key, self.base_url),
            )
        return self._cached_client

    def _build_request_params(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """统一整理 DashScope OpenAI 兼容接口参数。"""
        params: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stop": stop,
            "stream": stream,
        }
        params.update(kwargs)

        extra_body = dict(params.get("extra_body") or {})

        # DashScope 要求非流式请求必须显式关闭 thinking。
        if stream:
            extra_body.pop("enable_thinking", None)
        else:
            extra_body["enable_thinking"] = False

        if extra_body:
            params["extra_body"] = extra_body

        return params

    def _call_with_retry(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        """带重试的同步调用（指数退避）"""
        max_attempts = settings.llm_max_retries + 1
        last_exception = None

        for attempt in range(1, max_attempts + 1):
            try:
                response = self._client.chat.completions.create(
                    **self._build_request_params(
                        prompt=prompt,
                        stop=stop,
                        stream=False,
                        **kwargs,
                    )
                )
                answer = response.choices[0].message.content or ""
                return answer
            except Exception as e:
                last_exception = e
                if attempt < max_attempts:
                    wait_time = min(2 ** (attempt - 1), 10)  # 1s, 2s, 4s, ... 上限10s
                    logger.warning(
                        f"LLM 调用失败（第 {attempt}/{max_attempts} 次），"
                        f"{wait_time}s 后重试：{e}"
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"LLM 调用最终失败（已重试 {max_attempts - 1} 次）：{e}")

        raise last_exception  # type: ignore[misc]

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
            answer = self._call_with_retry(prompt, stop=stop, **kwargs)
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
                **self._build_request_params(
                    prompt=prompt,
                    stop=stop,
                    stream=True,
                    **kwargs,
                )
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
