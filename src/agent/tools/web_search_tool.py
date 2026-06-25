"""
网络搜索工具

支持两种搜索引擎，通过 WEB_SEARCH_PROVIDER 配置选择：
    bocha  —— 博查（国内直连，无需代理，推荐）
    tavily —— Tavily（需要代理/VPN，国外服务）

拿到搜索结果后用 LLM 生成答案。当本地知识库（RAG）检索不足时，
由 agent 调用此工具兜底。
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import requests

from config.settings import settings
from src.llm import QwenLLM
from src.utils import get_logger

logger = get_logger(__name__)


# 基于网络搜索结果生成答案的提示词
_WEB_ANSWER_PROMPT = """你是一个专业的问答助手。请根据以下网络搜索结果回答用户问题。

要求：
1. 基于搜索结果回答，不要编造
2. 如果搜索结果不足以回答，如实说明
3. 回答简洁、准确
4. 可以综合多条结果，但要客观

网络搜索结果：
{context}

用户问题：{question}

回答："""

BOCHA_URL = "https://api.bochaai.com/v1/web-search"


class WebSearchTool:
    """网络搜索工具（博查 / Tavily + LLM）"""

    name = "web_search"
    description = "在互联网上搜索实时信息并回答问题"

    def __init__(
        self,
        llm: Optional[QwenLLM] = None,
        provider: Optional[str] = None,
        max_results: int = settings.web_search_max_results,
        timeout: int = 30,
    ) -> None:
        self.llm = llm or QwenLLM()
        self.max_results = max_results
        self.timeout = timeout
        self.provider = (provider or settings.web_search_provider or "bocha").lower()

        if self.provider == "bocha":
            self.api_key = settings.bocha_api_key or os.environ.get("BOCHA_API_KEY", "")
            if not self.api_key:
                raise ValueError(
                    "BOCHA_API_KEY 未配置，无法使用博查搜索。"
                    "请在 .env 中设置 BOCHA_API_KEY。"
                )
        elif self.provider == "tavily":
            self.api_key = settings.tavily_api_key or os.environ.get("TAVILY_API_KEY", "")
            if not self.api_key:
                raise ValueError(
                    "TAVILY_API_KEY 未配置，无法使用 Tavily 搜索。"
                    "请在 .env 中设置 TAVILY_API_KEY。"
                )
        else:
            raise ValueError(f"未知的搜索引擎：{self.provider}（可选 bocha / tavily）")

        logger.info(
            f"WebSearchTool 初始化完成 | 引擎={self.provider} | max_results={max_results}"
        )

    # ──────────────────────────────────────────────────────────────────────────

    def search(self, question: str) -> Dict[str, Any]:
        """
        网络搜索并生成答案。

        Returns:
            {"answer": str, "sources": [str], "source_type": "网络"}
        """
        logger.info(f"网络搜索（{self.provider}）：{question[:60]}...")

        try:
            if self.provider == "bocha":
                results = self._search_bocha(question)
            else:
                results = self._search_tavily(question)
        except Exception as e:
            logger.error(f"网络搜索失败：{e}")
            return {"answer": f"网络搜索失败：{e}", "sources": [], "source_type": "网络"}

        if not results:
            return {"answer": "网络搜索未返回结果。", "sources": [], "source_type": "网络"}

        # 拼接搜索结果作为上下文
        context_parts, urls = [], []
        for i, r in enumerate(results, 1):
            title = r.get("title", "")
            content = r.get("content", "")
            url = r.get("url", "")
            context_parts.append(f"[{i}] {title}\n{content}")
            if url:
                urls.append(url)
        context = "\n\n".join(context_parts)

        # 用 LLM 基于搜索结果生成答案
        prompt = _WEB_ANSWER_PROMPT.format(context=context, question=question)
        try:
            answer = self.llm._call(prompt)
        except Exception as e:
            logger.error(f"LLM 生成网络答案失败：{e}")
            answer = f"已获取网络搜索结果，但生成答案时出错：{e}"

        logger.info(f"网络搜索完成 | 结果数={len(results)}")
        return {"answer": answer, "sources": urls, "source_type": "网络"}

    # ──────────────────────────────────────────────────────────────────────────
    # 博查搜索
    # ──────────────────────────────────────────────────────────────────────────

    def _search_bocha(self, question: str) -> List[Dict[str, str]]:
        """调用博查 Web Search API，返回标准化结果列表"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = json.dumps({
            "query": question,
            "summary": True,
            "count": self.max_results,
            "freshness": "noLimit",
        })
        resp = requests.post(
            BOCHA_URL, headers=headers, data=payload, timeout=self.timeout
        )
        resp.raise_for_status()
        data = resp.json()

        # 解析 data.webPages.value
        pages = (
            data.get("data", {})
            .get("webPages", {})
            .get("value", [])
        )
        results = []
        for p in pages:
            results.append({
                "title": p.get("name", ""),
                # 优先用 summary（更长），没有则用 snippet
                "content": p.get("summary") or p.get("snippet", ""),
                "url": p.get("url", ""),
            })
        return results

    # ──────────────────────────────────────────────────────────────────────────
    # Tavily 搜索
    # ──────────────────────────────────────────────────────────────────────────

    def _search_tavily(self, question: str) -> List[Dict[str, str]]:
        """调用 Tavily 搜索，返回标准化结果列表"""
        try:
            from tavily import TavilyClient
        except ImportError as e:
            raise ImportError("未安装 tavily-python，请运行：pip install tavily-python") from e

        client = TavilyClient(api_key=self.api_key)
        resp = client.search(
            query=question, max_results=self.max_results, search_depth="basic"
        )
        results = []
        for r in resp.get("results", []):
            results.append({
                "title": r.get("title", ""),
                "content": r.get("content", ""),
                "url": r.get("url", ""),
            })
        return results
