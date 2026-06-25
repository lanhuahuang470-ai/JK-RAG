"""
企业级 RAG FastAPI 服务
提供文档上传、索引管理、问答查询的 REST API
"""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from config.settings import settings
from src.chain import RAGChain
from src.agent import FallbackAgent
from src.agent.tools.rag_tool import RAGTool
from src.document_processor import DocumentProcessor
from src.embeddings import DashScopeEmbeddings
from src.llm import QwenLLM
from src.utils import get_logger
from src.vectorstore import FAISSVectorStore

logger = get_logger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# FastAPI 应用初始化
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="企业级 RAG 知识库问答系统",
    description=(
        "基于 LangChain + FAISS + Qwen3-Embedding + Qwen-Max 的企业级 RAG 系统\n\n"
        "**技术栈**\n"
        "- 向量检索：FAISS\n"
        "- 嵌入模型：Qwen3-Embedding（text-embedding-v4）\n"
        "- 大语言模型：qwen-max（stuff 问答链）\n"
        "- 文档处理：PyPDF2 + python-docx，chunk_size=1000, overlap=200"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────────────────────────
# 全局组件（延迟初始化）
# ──────────────────────────────────────────────────────────────────────────────

_components: Dict[str, Any] = {}


def get_components() -> Dict[str, Any]:
    """获取或初始化全局组件（单例）"""
    if not _components:
        logger.info("初始化全局组件...")
        settings.validate_api_key()

        embeddings = DashScopeEmbeddings()
        vector_store = FAISSVectorStore(embeddings=embeddings)
        # 尝试加载已有索引
        vector_store.load_index()

        llm = QwenLLM()
        rag_chain = RAGChain(vector_store=vector_store, llm=llm)
        agent = FallbackAgent(
            rag_tool=RAGTool(rag_chain=rag_chain),
            llm=llm,
        )
        doc_processor = DocumentProcessor()

        _components.update(
            {
                "embeddings": embeddings,
                "vector_store": vector_store,
                "llm": llm,
                "rag_chain": rag_chain,
                "agent": agent,
                "doc_processor": doc_processor,
            }
        )
        logger.info("全局组件初始化完成")
    return _components


# ──────────────────────────────────────────────────────────────────────────────
# Pydantic 模型
# ──────────────────────────────────────────────────────────────────────────────


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        description="检索 Top-K 数量，默认使用系统配置",
    )


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]
    retrieval_confidence: float
    source_scores: Dict[str, float]
    source_type: str
    route: str
    elapsed_ms: float


class DocumentUploadResponse(BaseModel):
    filename: str
    chunks_added: int
    total_chunks_in_index: int
    elapsed_ms: float


class IndexStatusResponse(BaseModel):
    is_loaded: bool
    total_chunks: int
    index_path: str
    embedding_model: str
    llm_model: str


class HealthResponse(BaseModel):
    status: str
    api_key_configured: bool
    index_loaded: bool
    version: str


# ──────────────────────────────────────────────────────────────────────────────
# API 路由
# ──────────────────────────────────────────────────────────────────────────────


@app.get("/", summary="服务根路径")
async def root() -> Dict[str, str]:
    return {
        "service": "企业级 RAG 知识库问答系统",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, summary="健康检查")
async def health_check() -> HealthResponse:
    """检查服务状态、API Key 配置、索引加载情况"""
    api_key_ok = bool(
        settings.dashscope_api_key or os.environ.get("DASHSCOPE_API_KEY")
    )
    index_loaded = False
    if _components:
        index_loaded = _components["vector_store"].is_loaded()

    return HealthResponse(
        status="healthy" if api_key_ok else "degraded",
        api_key_configured=api_key_ok,
        index_loaded=index_loaded,
        version="1.0.0",
    )


@app.post(
    "/documents/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="上传文档并构建/更新索引",
)
async def upload_document(file: UploadFile = File(...)) -> DocumentUploadResponse:
    """
    上传 PDF 或 DOCX 文件，自动提取文本、切分并写入 FAISS 索引。

    支持格式：`.pdf`, `.docx`
    """
    filename = file.filename or "unknown"
    suffix = Path(filename).suffix.lower()

    if suffix not in {".pdf", ".docx", ".doc"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"不支持的文件格式：{suffix}。支持：.pdf, .docx",
        )

    comps = get_components()
    t0 = time.time()

    # 将上传文件写入临时目录
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # 文档处理 → 切分
        docs = comps["doc_processor"].process_file(tmp_path)
        # 给 metadata 写入原始文件名（覆盖临时文件名）
        for doc in docs:
            doc.metadata["source"] = filename

        if not docs:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="文档内容为空或无法提取文本。",
            )

        # 写入 FAISS 索引（增量）
        comps["vector_store"].add_documents(docs)
        # 重建 RAG 链（确保 retriever 指向最新索引）
        comps["rag_chain"]._chain = None

        elapsed = (time.time() - t0) * 1000

        return DocumentUploadResponse(
            filename=filename,
            chunks_added=len(docs),
            total_chunks_in_index=comps["vector_store"].document_count,
            elapsed_ms=round(elapsed, 2),
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/query", response_model=QueryResponse, summary="知识库问答")
async def query(request: QueryRequest) -> QueryResponse:
    """
    通过已配置的 Agent 执行问答。

    流程：本地 RAG → 置信度判断 → 必要时网络搜索兜底。
    """
    comps = get_components()

    if not comps["vector_store"].is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="知识库索引尚未构建，请先上传文档。",
        )

    t0 = time.time()

    # 如果请求指定了 top_k，临时调整
    chain = comps["rag_chain"]
    agent = comps["agent"]
    original_k = chain.top_k
    if request.top_k is not None:
        chain.top_k = request.top_k
        chain._chain = None  # 重建 retriever

    try:
        result = agent.query(request.question)
    except Exception as e:
        logger.error(f"问答失败：{e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"问答服务异常：{str(e)}",
        )
    finally:
        if request.top_k is not None:
            chain.top_k = original_k
            chain._chain = None

    elapsed = (time.time() - t0) * 1000

    return QueryResponse(
        question=request.question,
        answer=result["answer"],
        sources=result["sources"],
        retrieval_confidence=result.get("confidence", 0.0),
        source_scores=result.get("source_scores", {}),
        source_type=result.get("source_type", "知识库"),
        route=result.get("route", "Agent"),
        elapsed_ms=round(elapsed, 2),
    )


@app.get("/index/status", response_model=IndexStatusResponse, summary="索引状态")
async def index_status() -> IndexStatusResponse:
    """返回当前 FAISS 索引状态"""
    comps = get_components()
    vs: FAISSVectorStore = comps["vector_store"]
    return IndexStatusResponse(
        is_loaded=vs.is_loaded(),
        total_chunks=vs.document_count,
        index_path=str(vs.index_path),
        embedding_model=settings.embedding_model,
        llm_model=settings.llm_model,
    )


@app.delete("/index/reset", summary="重置索引（危险操作）")
async def reset_index() -> JSONResponse:
    """清空所有向量索引（不可恢复），需重新上传文档"""
    comps = get_components()
    comps["vector_store"].reset_index()
    comps["rag_chain"]._chain = None
    logger.warning("索引已被 API 重置")
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"message": "索引已清空，请重新上传文档以重建索引。"},
    )


# ──────────────────────────────────────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        reload=False,
        log_level=settings.log_level.lower(),
    )
