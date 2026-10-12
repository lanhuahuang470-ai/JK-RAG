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
    # LLM 请求超时时间（秒）
    llm_timeout: float = Field(
        default=60.0,
        alias="LLM_TIMEOUT",
    )
    # LLM 最大重试次数（网络失败时自动重试）
    llm_max_retries: int = Field(
        default=2,
        alias="LLM_MAX_RETRIES",
    )
    # Embedding 请求超时时间（秒）
    embedding_timeout: float = Field(
        default=30.0,
        alias="EMBEDDING_TIMEOUT",
    )
    # Embedding 最大重试次数
    embedding_max_retries: int = Field(
        default=2,
        alias="EMBEDDING_MAX_RETRIES",
    )
    # 是否启用查询嵌入缓存（相同问题直接返回缓存，节省 API 调用）
    enable_embedding_cache: bool = Field(
        default=True,
        alias="ENABLE_EMBEDDING_CACHE",
    )
    # 是否启用检索结果缓存（相同查询直接返回缓存的检索结果）
    enable_retrieval_cache: bool = Field(
        default=True,
        alias="ENABLE_RETRIEVAL_CACHE",
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
    # 混合检索：向量 + BM25 关键词召回，RRF 融合（补齐条款编号类精确查询的短板）
    enable_hybrid_retrieval: bool = Field(
        default=True,
        alias="ENABLE_HYBRID_RETRIEVAL",
    )
    # 召回阶段数量（RRF 融合 / Rerank 精排前）
    retrieval_fetch_k: int = Field(
        default=20,
        alias="RETRIEVAL_FETCH_K",
    )
    # Rerank 精排：cross-encoder 对召回候选二次排序（只排小集合，换取精度）
    enable_rerank: bool = Field(
        default=True,
        alias="ENABLE_RERANK",
    )
    rerank_model: str = Field(
        default="gte-rerank-v2",
        alias="RERANK_MODEL",
    )

    # ── 日志 ───────────────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    class Config:
        env_file = ".env"
        populate_by_name = True
        # 忽略 .env 中未声明的冗余键，避免旧配置残留导致启动失败
        extra = "ignore"

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
