"""
evaluate_ragas_langchain.py

用途：
1. 读取评估数据集 jsonl。
2. 通过 API 或直接调用项目 RAG，收集 answer / contexts。
3. 保存可供 Ragas 评估使用的 records jsonl。
4. 可选执行 Ragas 评估，输出明细 csv 与汇总 json。
"""

import argparse
import json
import os
import sys
import types
from pathlib import Path
from typing import Any, Dict, List

import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RAGAS_ENV_FILES = [
    PROJECT_ROOT / ".env.ragas",
    Path.cwd() / ".env.ragas",
]
for env_file in RAGAS_ENV_FILES:
    if env_file.is_file():
        load_dotenv(env_file, override=False)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_contexts(raw: Any) -> List[str]:
    """兼容不同返回格式，统一转成 List[str]。"""
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        contexts: List[str] = []
        for item in raw:
            if isinstance(item, str):
                contexts.append(item)
            elif isinstance(item, dict):
                contexts.append(
                    item.get("page_content")
                    or item.get("content")
                    or item.get("text")
                    or item.get("snippet")
                    or json.dumps(item, ensure_ascii=False)
                )
            else:
                contexts.append(str(item))
        return contexts
    return [str(raw)]


def ask_rag_api(question: str, url: str, timeout: int = 120) -> Dict[str, Any]:
    payload_candidates = [
        {"query": question},
        {"question": question},
        {"message": question},
    ]

    last_error = None
    resp_json = None

    for payload in payload_candidates:
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            if resp.status_code < 400:
                resp_json = resp.json()
                break
            last_error = f"HTTP {resp.status_code}: {resp.text[:500]}"
        except Exception as exc:  # pragma: no cover
            last_error = repr(exc)

    if resp_json is None:
        raise RuntimeError(f"调用 RAG 接口失败：{last_error}")

    answer = (
        resp_json.get("answer")
        or resp_json.get("result")
        or resp_json.get("response")
        or resp_json.get("output")
        or ""
    )

    raw_contexts = (
        resp_json.get("contexts")
        or resp_json.get("source_documents")
        or resp_json.get("documents")
        or resp_json.get("docs")
    )
    contexts = normalize_contexts(raw_contexts)

    if not answer:
        raise ValueError(
            f"接口返回中没有找到 answer/result/response/output 字段：{resp_json}"
        )

    if not contexts:
        raise ValueError(
            "接口没有返回 contexts/source_documents/documents/docs。\n"
            "Ragas 检索评估必须使用实际检索到的 chunk。"
        )

    return {
        "answer": str(answer),
        "contexts": contexts,
        "sources": resp_json.get("sources", []),
        "source_type": resp_json.get("source_type", ""),
        "route": resp_json.get("route", ""),
        "retrieval_confidence": resp_json.get("retrieval_confidence"),
        "source_scores": resp_json.get("source_scores", {}),
        "timings_ms": resp_json.get("timings_ms", {}),
    }


def ask_rag_direct(question: str) -> Dict[str, Any]:
    """直接调用项目内部 RAG。"""
    from config.settings import settings
    from src.rag.rag_chain import RAGChain
    from src.rag.embeddings import DashScopeEmbeddings
    from src.llm import QwenLLM
    from src.rag.vectorstore import FAISSVectorStore

    settings.validate_api_key()

    embeddings = DashScopeEmbeddings()
    vector_store = FAISSVectorStore(embeddings=embeddings)
    if not vector_store.load_index():
        raise RuntimeError(
            "FAISS 索引未加载，请先执行 `python run.py index ...` 构建索引。"
        )

    llm = QwenLLM()
    rag = RAGChain(
        vector_store=vector_store,
        llm=llm,
        enable_query_rewrite=settings.enable_query_rewrite,
    )

    result = rag.query(question)
    answer = str(result.get("answer", ""))
    contexts = [doc.page_content for doc in result.get("source_documents", [])]

    if not answer:
        raise ValueError("RAGChain 返回结果中缺少 answer。")
    if not contexts:
        raise ValueError("RAGChain 返回结果中缺少 source_documents，无法用于 Ragas 评估。")

    return {
        "answer": answer,
        "contexts": contexts,
        "sources": result.get("sources", []),
        "source_type": "知识库",
        "route": "direct_rag",
        "retrieval_confidence": result.get("retrieval_confidence"),
        "source_scores": result.get("source_scores", {}),
        "timings_ms": result.get("timings_ms", {}),
    }


def build_output_record(sample: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": sample.get("id"),
        "question": sample["question"],
        "answer": result["answer"],
        "contexts": result["contexts"],
        "ground_truth": sample.get("ground_truth") or sample.get("reference"),
        "reference": sample.get("reference") or sample.get("ground_truth"),
        "source_file": sample.get("source_file"),
        "source_section": sample.get("source_section"),
        "question_type": sample.get("question_type"),
        "expected_source_type": sample.get("expected_source_type", ""),
        "expected_route": sample.get("expected_route", ""),
        "source_type": result.get("source_type", ""),
        "route": result.get("route", ""),
        "retrieval_confidence": result.get("retrieval_confidence"),
        "sources": result.get("sources", []),
        "source_scores": result.get("source_scores", {}),
        "timings_ms": result.get("timings_ms", {}),
    }


def build_route_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    routed = [
        record
        for record in records
        if record.get("expected_source_type") or record.get("expected_route")
    ]
    if not routed:
        return {}

    matched = 0
    for record in routed:
        expected_source = (record.get("expected_source_type") or "").strip()
        expected_route = (record.get("expected_route") or "").strip()
        actual_source = (record.get("source_type") or "").strip()
        actual_route = (record.get("route") or "").strip()

        ok = False
        if expected_source and actual_source and expected_source == actual_source:
            ok = True
        if expected_route and actual_route and expected_route == actual_route:
            ok = True
        if ok:
            matched += 1

    return {
        "route_eval_count": len(routed),
        "route_match_count": matched,
        "route_accuracy": round(matched / len(routed), 4) if routed else None,
    }


def collect_rag_outputs(args) -> List[Dict[str, Any]]:
    testset = load_jsonl(args.dataset)
    if args.limit:
        testset = testset[: args.limit]

    outputs: List[Dict[str, Any]] = []
    for idx, sample in enumerate(testset, start=1):
        question = sample["question"]
        print(f"[{idx}/{len(testset)}] Query: {question}")

        if args.mode == "api":
            result = ask_rag_api(question, args.url, timeout=args.timeout)
        elif args.mode == "direct":
            result = ask_rag_direct(question)
        elif args.mode == "debug_golden_contexts":
            result = {
                "answer": sample["ground_truth"],
                "contexts": sample.get("evidence_contexts", []),
                "sources": [sample.get("source_file")] if sample.get("source_file") else [],
                "source_type": sample.get("expected_source_type", "知识库"),
                "route": sample.get("expected_route", "debug_golden_contexts"),
                "retrieval_confidence": None,
                "source_scores": {},
                "timings_ms": {},
            }
        else:
            raise ValueError(f"未知 mode: {args.mode}")

        outputs.append(build_output_record(sample, result))

    save_jsonl(outputs, args.out)
    print(f"\n已保存 Ragas 输入文件：{args.out}")
    return outputs


def build_evaluator_llm_and_embeddings(args):
    if args.judge_backend == "ollama":
        from langchain_ollama import ChatOllama

        judge_llm = ChatOllama(
            model=args.judge_model,
            temperature=0,
            num_predict=args.judge_num_predict,
            base_url=args.ollama_base_url,
        )
    elif args.judge_backend == "openai_compatible":
        from langchain_openai import ChatOpenAI

        api_key = (
            args.api_key
            or os.getenv("RAGAS_API_KEY")
            or os.getenv("OPENAI_API_KEY", "EMPTY")
        )
        api_base = (
            args.api_base
            or os.getenv("RAGAS_API_BASE")
            or os.getenv("OPENAI_API_BASE")
        )

        judge_llm = ChatOpenAI(
            model=args.judge_model,
            temperature=0,
            api_key=api_key,
            base_url=api_base,
            extra_body={"enable_thinking": False} if args.disable_thinking else None,
        )
    else:
        raise ValueError("judge_backend 只能是 ollama 或 openai_compatible")

    if args.embedding_model.startswith("dashscope"):
        # 本地 bge-m3 不可用时的备选：复用项目 DashScope 嵌入做评估嵌入
        # （注意：answer_relevancy 依赖嵌入计算相似度，与 bge-m3 基线尺度略有差异；
        #  其余三项指标由 LLM judge 主导，不受影响）
        from src.rag.embeddings import DashScopeEmbeddings

        judge_embeddings = DashScopeEmbeddings()
    else:
        from langchain_huggingface import HuggingFaceEmbeddings

        judge_embeddings = HuggingFaceEmbeddings(
            model_name=args.embedding_model,
            encode_kwargs={"normalize_embeddings": True},
        )
    return judge_llm, judge_embeddings


def ensure_ragas_import_compat() -> None:
    """
    兼容部分 ragas / langchain-community 版本组合。
    某些 ragas 导入时会尝试引入已移除的 vertexai 模块，这里提供占位。
    """
    module_name = "langchain_community.chat_models.vertexai"
    if module_name in sys.modules:
        return

    try:
        __import__(module_name)
        return
    except ModuleNotFoundError:
        pass

    shim = types.ModuleType(module_name)

    class ChatVertexAI:  # pragma: no cover
        pass

    shim.ChatVertexAI = ChatVertexAI
    sys.modules[module_name] = shim


def run_ragas(args, records: List[Dict[str, Any]] = None) -> None:
    import pandas as pd
    from datasets import Dataset

    if records is None:
        source_path = args.records_in or args.out
        records = load_jsonl(source_path)

    dataset = Dataset.from_list(
        [
            {
                "question": record["question"],
                "answer": record["answer"],
                "contexts": record["contexts"],
                "ground_truth": record.get("ground_truth") or record.get("reference"),
            }
            for record in records
        ]
    )

    judge_llm, judge_embeddings = build_evaluator_llm_and_embeddings(args)

    ensure_ragas_import_compat()

    from ragas import evaluate

    try:
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    except Exception:
        from ragas.metrics import (
            Faithfulness,
            LLMContextPrecisionWithReference,
            LLMContextRecall,
            ResponseRelevancy,
        )

        metrics = [
            Faithfulness(),
            ResponseRelevancy(),
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
        ]

    print("\n开始 Ragas 评估……")
    result = evaluate(
        dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_embeddings,
        raise_exceptions=False,
        show_progress=True,
    )

    df = result.to_pandas()
    df.to_csv(args.scores, index=False, encoding="utf-8-sig")
    print(f"明细分数已保存：{args.scores}")

    summary: Dict[str, Any] = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            summary[col] = float(df[col].mean(skipna=True))

    summary.update(build_route_summary(records))

    with open(args.summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"汇总分数已保存：{args.summary}")
    print("\n平均分：")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ragas_eval_50.jsonl", help="评估数据集 jsonl")
    parser.add_argument(
        "--mode",
        choices=["api", "direct", "debug_golden_contexts"],
        default="direct",
    )
    parser.add_argument("--url", default="http://127.0.0.1:8000/query", help="FastAPI /query 地址")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--limit", type=int, default=0, help="先小批量测试，比如 5；0 表示全量")
    parser.add_argument("--out", default="ragas_eval_inputs_with_answers.jsonl")
    parser.add_argument("--run-ragas", action="store_true", help="是否执行 Ragas 评估")

    parser.add_argument(
        "--judge-backend",
        choices=["ollama", "openai_compatible"],
        default="ollama",
    )
    parser.add_argument("--judge-model", default="qwen2.5:7b-instruct", help="评估 LLM")
    parser.add_argument("--judge-num-predict", type=int, default=1024)
    parser.add_argument("--ollama-base-url", default="http://127.0.0.1:11434")
    parser.add_argument(
        "--api-key",
        default="",
        help="OpenAI 兼容 API 密钥；不填则读取 RAGAS_API_KEY / OPENAI_API_KEY",
    )
    parser.add_argument(
        "--api-base",
        default="",
        help="OpenAI 兼容 API Base；不填则读取 RAGAS_API_BASE / OPENAI_API_BASE",
    )
    parser.add_argument(
        "--embedding-model",
        default="./models/bge-m3",
        help="评估用 embedding 模型路径或名称",
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="对部分 OpenAI 兼容模型关闭 thinking",
    )
    parser.add_argument("--scores", default="ragas_scores.csv")
    parser.add_argument("--summary", default="ragas_summary.json")
    parser.add_argument(
        "--records-in",
        default="",
        help="直接读取已采样好的 RAG 输出 jsonl，跳过 collect 阶段",
    )

    args = parser.parse_args()

    if args.records_in:
        records = load_jsonl(args.records_in)
        print(f"\n已读取现成 RAG 输出文件：{args.records_in}")
    else:
        records = collect_rag_outputs(args)

    if args.run_ragas:
        run_ragas(args, records)
    else:
        print("\n未运行 Ragas 评估。确认 RAG 输出文件无误后，再加 --run-ragas 执行。")


if __name__ == "__main__":
    main()
