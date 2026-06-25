#!/usr/bin/env python3
"""
命令行入口：快速测试 RAG 系统（无需启动 HTTP 服务）

使用示例：
    # 上传并索引文档（指定具体文件）
    python run.py index --files docs/report.pdf docs/manual.docx

    # 或自动扫描整个文件夹下的所有 PDF/Word
    python run.py index --dir data

    # 提问
    python run.py query "公司的退款政策是什么？"

    # 启动 API 服务
    python run.py serve
"""
import argparse
import sys
from pathlib import Path


def cmd_index(args: argparse.Namespace) -> None:
    """处理文档并构建/更新 FAISS 索引"""
    from config.settings import settings
    settings.validate_api_key()

    from src.document_processor import get_processor
    from src.embeddings import DashScopeEmbeddings
    from src.vectorstore import FAISSVectorStore

    # 根据 .env 里的 CHUNK_STRATEGY 自动选择切分策略
    processor = get_processor(getattr(args, "strategy", None))
    embeddings = DashScopeEmbeddings()
    vs = FAISSVectorStore(embeddings=embeddings)

    # 收集待处理文件：支持 --files（显式列文件）和 --dir（扫描整个文件夹）
    file_paths: list[Path] = []

    if args.dir:
        scan_dir = Path(args.dir)
        if not scan_dir.is_dir():
            print(f"[错误] 文件夹不存在：{scan_dir}")
            sys.exit(1)
        # 自动找出该文件夹下所有 PDF / Word 文档
        for ext in ("*.pdf", "*.docx", "*.doc"):
            file_paths.extend(scan_dir.glob(ext))
        if not file_paths:
            print(f"[错误] 文件夹 {scan_dir} 里没有找到 PDF / Word 文档")
            sys.exit(1)
        print(f"📂 扫描文件夹 {scan_dir}，找到 {len(file_paths)} 个文档：")
        for fp in file_paths:
            print(f"   - {fp.name}")

    if args.files:
        for f in args.files:
            fp = Path(f)
            if not fp.exists():
                print(f"[错误] 文件不存在：{fp}")
                sys.exit(1)
            file_paths.append(fp)

    if not file_paths:
        # 既没指定 --files 也没指定 --dir，默认扫描 data 文件夹
        default_dir = Path("data")
        if default_dir.is_dir():
            print(f"📂 未指定文件，默认扫描 {default_dir} 文件夹...")
            for ext in ("*.pdf", "*.docx", "*.doc"):
                file_paths.extend(default_dir.glob(ext))
            for fp in file_paths:
                print(f"   - {fp.name}")
        if not file_paths:
            print("[错误] 请用 --files 指定文件，或用 --dir 指定文件夹，"
                  "或把文档放进 data 文件夹")
            sys.exit(1)

    docs = processor.process_files(file_paths)
    print(f"✅ 共处理 {len(docs)} 个 chunk，开始写入向量库...")

    if args.rebuild:
        vs.build_index(docs)
    else:
        vs.add_documents(docs)

    print(f"✅ 索引更新完成 | 当前共 {vs.document_count} 个向量")


def cmd_query(args: argparse.Namespace) -> None:
    """执行 RAG 问答"""
    from config.settings import settings
    settings.validate_api_key()

    from src.chain import RAGChain
    from src.embeddings import DashScopeEmbeddings
    from src.llm import QwenLLM
    from src.vectorstore import FAISSVectorStore

    embeddings = DashScopeEmbeddings()
    vs = FAISSVectorStore(embeddings=embeddings)
    if not vs.load_index():
        print("❌ 索引未构建，请先运行 `python run.py index --files <文件路径>`")
        sys.exit(1)

    llm = QwenLLM()
    # --rewrite 命令行开关：指定则覆盖配置
    enable_rewrite = True if getattr(args, "rewrite", False) else None
    chain = RAGChain(vector_store=vs, llm=llm, enable_query_rewrite=enable_rewrite)

    # ── 网络兜底分支：--web 或配置开启时，走 FallbackAgent ──────────────────
    use_web = getattr(args, "web", False) or settings.enable_web_fallback
    if use_web:
        from src.agent import FallbackAgent
        from src.agent.tools.rag_tool import RAGTool

        agent = FallbackAgent(
            rag_tool=RAGTool(rag_chain=chain),
            llm=llm,
            enable_web_fallback=True,
        )
        ares = agent.query(args.question)
        print("\n" + "=" * 60)
        print(f"❓ 问题：{args.question}")
        print(f"💡 回答：\n{ares['answer']}")
        print(f"📄 来源：{', '.join(ares.get('sources', [])) or '无'}")
        print(f"🧭 决策路径：{ares.get('route', '')}")
        print(f"📍 答案来源类型：{ares.get('source_type', '')}")
        print("=" * 60)
        return

    result = chain.query(args.question)

    print("\n" + "=" * 60)
    print(f"❓ 问题：{result['question']}")
    # 若启用了查询改写，显示改写后的问题
    if result.get("rewritten_query") and result["rewritten_query"] != result["question"]:
        print(f"🔍 改写后：{result['rewritten_query']}")
    print(f"💡 回答：\n{result['answer']}")
    print(f"📄 来源：{', '.join(result['sources']) or '无'}")

    # 检索置信度（相似度分数，0~1，越高越相关）
    conf = result.get("retrieval_confidence", 0.0)
    print(f"🎯 检索置信度：{conf}  （0~1，越接近 1 表示资料越相关）")
    scores = result.get("source_scores", {})
    if scores:
        print("   各来源相似度：")
        for src, sc in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            print(f"     - {src}: {sc}")
    print("=" * 60)


def cmd_serve(args: argparse.Namespace) -> None:
    """启动 FastAPI HTTP 服务"""
    import uvicorn
    from config.settings import settings

    uvicorn.run(
        "src.api.main:app",
        host=args.host or settings.api_host,
        port=args.port or settings.api_port,
        reload=args.reload,
        log_level=settings.log_level.lower(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="企业级 RAG 系统命令行工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # index 子命令
    index_parser = subparsers.add_parser("index", help="处理文档并构建向量索引")
    index_parser.add_argument("--files", nargs="+", help="文件路径列表（可指定多个）")
    index_parser.add_argument("--dir", help="文件夹路径，自动扫描其中所有 PDF/Word 文档")
    index_parser.add_argument(
        "--strategy",
        choices=["recursive", "semantic", "hierarchical"],
        default=None,
        help="切分策略（不填则用 .env 里的 CHUNK_STRATEGY）",
    )
    index_parser.add_argument(
        "--rebuild", action="store_true", help="全量重建索引（默认为增量更新）"
    )

    # query 子命令
    query_parser = subparsers.add_parser("query", help="向知识库提问")
    query_parser.add_argument("question", help="用户问题")
    query_parser.add_argument(
        "--rewrite", action="store_true",
        help="启用查询改写（检索前用 LLM 优化问题）",
    )
    query_parser.add_argument(
        "--web", action="store_true",
        help="启用网络兜底（RAG 检索不足时转 Tavily 网络搜索）",
    )

    # serve 子命令
    serve_parser = subparsers.add_parser("serve", help="启动 FastAPI 服务")
    serve_parser.add_argument("--host", default=None, help="监听地址")
    serve_parser.add_argument("--port", type=int, default=None, help="监听端口")
    serve_parser.add_argument("--reload", action="store_true", help="开发模式热重载")

    args = parser.parse_args()

    if args.command == "index":
        cmd_index(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "serve":
        cmd_serve(args)


if __name__ == "__main__":
    main()
