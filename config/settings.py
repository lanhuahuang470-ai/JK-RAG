"""
全局配置模块
从环境变量或 .env 文件加载所有配置
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv

# 加载 .env 文件（开发环境使用；生产环境直接注入环境变量）
load_dotenv()


class Settings(BaseSettings):
    """企业级 RAG 全局配置"""

    # ── DashScope ──────────────────────────────────────────────────────────────
    dashscope_api_key: str = Field(
        default="",
        alias="DASHSCOPE_API_KEY",
        description="阿里云百炼平台 API Key",
    )

    # ── 嵌入模型 ───────────────────────────────────────────────────────────────
    embedding_model: str = Field(
        default="text-embedding-v4",
        alias="EMBEDDING_MODEL",
    )
    embedding_dimension: int = Field(
        default=2048,
        alias="EMBEDDING_DIMENSION",
    )

    # ── 大语言模型 ─────────────────────────────────────────────────────────────
    llm_model: str = Field(
        default="qwen-max",
        alias="LLM_MODEL",
    )
    llm_temperature: float = Field(
        default=0.1,
        alias="LLM_TEMPERATURE",
    )
    llm_max_tokens: int = Field(
        default=2048,
        alias="LLM_MAX_TOKENS",
    )

    # ── 文档切分 ───────────────────────────────────────────────────────────────
    chunk_size: int = Field(
        default=1000,
        alias="CHUNK_SIZE",
    )
    chunk_overlap: int = Field(
        default=200,
        alias="CHUNK_OVERLAP",
    )
    # 切分策略：recursive（递归，默认）/ semantic（语义）/ hierarchical（层次）
    chunk_strategy: str = Field(
        default="recursive",
        alias="CHUNK_STRATEGY",
    )

    # ── 向量库 ─────────────────────────────────────────────────────────────────
    faiss_index_path: str = Field(
        default="data/vectorstore/faiss_index",
        alias="FAISS_INDEX_PATH",
    )

    # ── FastAPI ────────────────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    api_workers: int = Field(default=1, alias="API_WORKERS")

    # ── 检索 ───────────────────────────────────────────────────────────────────
    retrieval_top_k: int = Field(
        default=5,
        alias="RETRIEVAL_TOP_K",
    )
    retrieval_score_threshold: float = Field(
        default=0.5,
        alias="RETRIEVAL_SCORE_THRESHOLD",
    )
    # 是否启用查询改写（检索前用 LLM 优化问题，提升命中率）
    enable_query_rewrite: bool = Field(
        default=False,
        alias="ENABLE_QUERY_REWRITE",
    )

    # ── Agent：RAG + 网络兜底 ────────────────────────────────────────────────
    # 搜索引擎提供方：bocha（博查，国内直连）/ tavily（需代理）
    web_search_provider: str = Field(default="bocha", alias="WEB_SEARCH_PROVIDER")
    # Tavily 搜索 API Key
    tavily_api_key: str = Field(default="", alias="TAVILY_API_KEY")
    # 博查 Bocha 搜索 API Key
    bocha_api_key: str = Field(default="", alias="BOCHA_API_KEY")
    # 是否启用网络兜底（RAG 不足时转网络搜索）
    enable_web_fallback: bool = Field(default=False, alias="ENABLE_WEB_FALLBACK")
    # 高置信度阈值：≥ 此值直接用 RAG
    fallback_high_threshold: float = Field(
        default=0.65, alias="FALLBACK_HIGH_THRESHOLD"
    )
    # 低置信度阈值：≤ 此值直接转网络
    fallback_low_threshold: float = Field(
        default=0.45, alias="FALLBACK_LOW_THRESHOLD"
    )
    # 网络搜索返回结果数
    web_search_max_results: int = Field(default=5, alias="WEB_SEARCH_MAX_RESULTS")

    # ── 日志 ───────────────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    class Config:
        env_file = ".env"
        populate_by_name = True

    @property
    def faiss_index_dir(self) -> Path:
        """向量库目录（自动创建）"""
        path = Path(self.faiss_index_path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def validate_api_key(self) -> None:
        """运行时校验必填项"""
        if not self.dashscope_api_key:
            raise ValueError(
                "DASHSCOPE_API_KEY 未设置！"
                "请在 .env 文件或环境变量中配置该值。"
            )


# 全局单例
settings = Settings()
