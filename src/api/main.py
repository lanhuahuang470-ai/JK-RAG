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
from src.llm import QwenLLM
from src.api.conversation import ChatMessage, conversation_manager
from src.rag import RAGChain, DashScopeEmbeddings, DocumentProcessor, FAISSVectorStore
from src.utils import get_logger

logger = get_logger(__name__)

# ── FastAPI 应用初始化 ──────────────────────────────────────────────────

app = FastAPI(
    title="企业级 RAG 知识库问答系统",
    description=(
        f"基于 LangChain + FAISS + {settings.embedding_model} + {settings.llm_model} 的企业级 RAG 系统\n\n"
        "**技术栈**\n"
        "- 向量检索：FAISS\n"
        f"- 嵌入模型：{settings.embedding_model}\n"
        f"- 大语言模型：{settings.llm_model}（stuff 问答链）\n"
        "- 文档处理：层次切分（PyPDF2 + python-docx）\n"
        "- 多轮对话：查询改写 + 会话管理"
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

# ── 全局组件（延迟初始化）──────────────────────────────────────────────

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
        doc_processor = DocumentProcessor()

        _components.update(
            {
                "embeddings": embeddings,
                "vector_store": vector_store,
                "llm": llm,
                "rag_chain": rag_chain,
                "doc_processor": doc_processor,
            }
        )
        logger.info("全局组件初始化完成")
    return _components


# ── Pydantic 模型 ──────────────────────────────────────────────────────


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        description="检索 Top-K 数量，默认使用系统配置",
    )
    conversation_id: Optional[str] = Field(
        default=None,
        description="会话 ID，传入则使用该会话的历史上下文进行多轮对话",
    )


class ConversationQueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        description="检索 Top-K 数量，默认使用系统配置",
    )


class ConversationCreateResponse(BaseModel):
    id: str
    title: str
    created_at: float


class ConversationItem(BaseModel):
    id: str
    title: str
    created_at: float
    updated_at: float
    message_count: int


class MessageItem(BaseModel):
    role: str
    content: str
    sources: List[str]
    source_type: Optional[str]
    route: Optional[str]
    timestamp: float


class ConversationDetailResponse(BaseModel):
    id: str
    title: str
    created_at: float
    updated_at: float
    messages: List[MessageItem]


class QueryResponse(BaseModel):
    question: str
    answer: str
    contexts: List[str]
    sources: List[str]
    retrieval_confidence: float
    source_scores: Dict[str, float]
    source_type: str
    route: str
    elapsed_ms: float
    timings_ms: Dict[str, float]


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


# ── API 路由 ───────────────────────────────────────────────────────────


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
        # 上传完成后强制持久化，确保数据不丢
        comps["vector_store"].flush()
        # 清空检索缓存（索引已更新）
        from src.rag.rag_chain import clear_retrieval_cache
        clear_retrieval_cache()
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
    执行 RAG 知识库问答。

    流程：查询改写（多轮时）→ FAISS 检索 → Stuff 拼接上下文 → LLM 生成。
    传入 conversation_id 则使用该会话的历史上下文进行多轮对话。
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
    original_k = chain.top_k
    if request.top_k is not None:
        chain.top_k = request.top_k
        chain._chain = None  # 重建 retriever

    # 获取历史消息（如果有会话 ID）
    chat_history = None
    if request.conversation_id:
        conv = conversation_manager.get(request.conversation_id)
        if conv is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"会话 {request.conversation_id} 不存在",
            )
        chat_history = [m.to_dict() for m in conv.messages]

    try:
        result = chain.query(request.question, chat_history=chat_history)
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

    # 如果有会话 ID，把消息追加到会话中
    if request.conversation_id:
        conv = conversation_manager.get(request.conversation_id)
        if conv is not None:
            conv.add_message(ChatMessage(role="user", content=request.question))
            conv.add_message(
                ChatMessage(
                    role="assistant",
                    content=result["answer"],
                    sources=result["sources"],
                    source_type="知识库",
                    route="RAG",
                )
            )

    return QueryResponse(
        question=request.question,
        answer=result["answer"],
        contexts=[doc.page_content for doc in result.get("source_documents", [])],
        sources=result["sources"],
        retrieval_confidence=result.get("retrieval_confidence", 0.0),
        source_scores=result.get("source_scores", {}),
        source_type="知识库",
        route="RAG",
        elapsed_ms=round(elapsed, 2),
        timings_ms=result.get("timings_ms", {}),
    )


# ── 会话管理 API ────────────────────────────────────────────────────────


@app.post(
    "/conversations",
    response_model=ConversationCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="创建新会话",
)
async def create_conversation() -> ConversationCreateResponse:
    """创建一个新的多轮对话会话"""
    conv = conversation_manager.create()
    logger.info(f"创建新会话：{conv.id}")
    return ConversationCreateResponse(
        id=conv.id,
        title=conv.title,
        created_at=conv.created_at,
    )


@app.get(
    "/conversations",
    response_model=List[ConversationItem],
    summary="获取会话列表",
)
async def list_conversations() -> List[ConversationItem]:
    """获取所有会话列表，按更新时间倒序"""
    convs = conversation_manager.list()
    return [ConversationItem(**c) for c in convs]


@app.get(
    "/conversations/{conv_id}",
    response_model=ConversationDetailResponse,
    summary="获取会话详情",
)
async def get_conversation(conv_id: str) -> ConversationDetailResponse:
    """获取指定会话的详情（含历史消息）"""
    conv = conversation_manager.get(conv_id)
    if conv is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"会话 {conv_id} 不存在",
        )
    data = conv.to_dict(with_messages=True)
    return ConversationDetailResponse(**data)


@app.post(
    "/conversations/{conv_id}/messages",
    response_model=QueryResponse,
    summary="在指定会话中发送消息",
)
async def send_message(conv_id: str, request: ConversationQueryRequest) -> QueryResponse:
    """
    在指定会话中发送消息，自动带上历史上下文。
    会话不存在时返回 404。
    """
    # 复用 query 接口的逻辑，直接用 conversation_id
    full_request = QueryRequest(
        question=request.question,
        top_k=request.top_k,
        conversation_id=conv_id,
    )
    return await query(full_request)


@app.delete(
    "/conversations/{conv_id}",
    summary="删除会话",
)
async def delete_conversation(conv_id: str) -> JSONResponse:
    """删除指定会话"""
    ok = conversation_manager.delete(conv_id)
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"会话 {conv_id} 不存在",
        )
    logger.info(f"删除会话：{conv_id}")
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"message": f"会话 {conv_id} 已删除"},
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
    # 清空检索缓存
    from src.rag.rag_chain import clear_retrieval_cache
    clear_retrieval_cache()
    logger.warning("索引已被 API 重置")
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"message": "索引已清空，请重新上传文档以重建索引。"},
    )


# ── 入口 ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        reload=False,
        log_level=settings.log_level.lower(),
    )
