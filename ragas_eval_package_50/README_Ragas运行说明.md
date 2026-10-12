# Ragas评估运行说明（精选50条版）

本包用于你自己的 LangChain + FAISS + RAGChain 系统做 Ragas 评估。

## 文件说明

- `ragas_eval_50.jsonl`：精选50条评估数据，推荐作为脚本输入。
- `ragas_eval_50.csv`：同一批数据的CSV版，方便人工查看。
- `ragas_eval_50_人工评估数据说明.md`：人工可读版说明和50条明细。
- `evaluate_ragas_langchain.py`：评估脚本。

## 重要说明

`evidence_contexts` 是人工从原文整理的依据片段，只用于人工检查或 `debug_golden_contexts` 模式。

正式评估时，Ragas里的 `contexts` 必须来自你自己的RAG检索器实际返回的chunk，不能直接拿人工答案片段代替，否则只能测试Ragas流程，不能真实评估你的FAISS检索质量。

你的 `/query` 接口最好返回：

```json
{
  "answer": "你的RAG答案",
  "contexts": ["FAISS检索出来的chunk1", "FAISS检索出来的chunk2"]
}
```

脚本也兼容以下字段：

- 答案字段：`answer` / `result` / `response` / `output`
- 上下文字段：`contexts` / `source_documents` / `documents` / `docs`

## 运行步骤

### 1. 解压并放入项目根目录

例如：

```powershell
E:\JK-RAG\
  run.py
  ragas_eval_50.jsonl
  evaluate_ragas_langchain.py
```

### 2. 激活虚拟环境

```powershell
cd E:\JK-RAG
.\.venv\Scripts\activate
```

### 3. 安装依赖

```powershell
pip install ragas datasets pandas requests langchain-ollama langchain-huggingface sentence-transformers
```

如果你用OpenAI兼容接口做评估模型，再装：

```powershell
pip install langchain-openai
```

如果你想用 DeepSeek、阿里云百炼或其他 OpenAI 兼容 API 做 judge，脚本现在支持三种传法：

```powershell
# 方式一：评估专用 .env.ragas（推荐）
# 放在项目根目录 E:\JK-RAG\.env.ragas
RAGAS_API_KEY=你的API Key
RAGAS_API_BASE=你的API Base URL
```

```powershell
# 方式二：环境变量
$env:OPENAI_API_KEY="你的API Key"
$env:OPENAI_API_BASE="你的API Base URL"
```

```powershell
# 方式三：命令行参数
--api-key "你的API Key" --api-base "你的API Base URL"
```

脚本读取优先级：
1. 命令行参数 `--api-key` / `--api-base`
2. `.env.ragas` 里的 `RAGAS_API_KEY` / `RAGAS_API_BASE`
3. 系统环境变量 `OPENAI_API_KEY` / `OPENAI_API_BASE`

### 4. 纯 RAG 基线评估推荐做法

优先使用 `direct` 模式，直接调用项目内部 `RAGChain`，这样不会经过 FastAPI `/query`，也不会把 Agent 路由逻辑混进纯 RAG 基线。
当前脚本默认模式也已经改为 `direct`。

先确认：

- 已经构建好本地索引
- 根目录 `.env` 中建议设置 `ENABLE_WEB_FALLBACK=false`

例如先建索引：

```powershell
python run.py index --dir data
```

### 5. 先只跑5条，不打分，只收集纯 RAG 输出

```powershell
python .\ragas_eval_package_50\evaluate_ragas_langchain.py --dataset .\ragas_eval_package_50\ragas_eval_50.jsonl --mode direct --limit 5 --out .\ragas_eval_package_50\ragas_eval_inputs_with_answers_5.jsonl
```

打开生成的 `ragas_eval_inputs_with_answers_5.jsonl`，确认每条都有：

```json
{
  "question": "...",
  "answer": "...",
  "contexts": ["..."],
  "ground_truth": "..."
}
```

### 6. 小批量Ragas评分

如果你本地有Ollama模型，例如 `qwen2.5:7b-instruct`：

```powershell
python .\ragas_eval_package_50\evaluate_ragas_langchain.py --dataset .\ragas_eval_package_50\ragas_eval_50.jsonl --mode direct --limit 5 --out .\ragas_eval_package_50\ragas_eval_inputs_with_answers_5.jsonl --run-ragas --judge-backend ollama --judge-model qwen2.5:7b-instruct --embedding-model .\models\bge-m3 --scores .\ragas_eval_package_50\ragas_scores_5.csv --summary .\ragas_eval_package_50\ragas_summary_5.json
```

### 7. 全量50条评分

```powershell
python .\ragas_eval_package_50\evaluate_ragas_langchain.py --dataset .\ragas_eval_package_50\ragas_eval_50.jsonl --mode direct --out .\ragas_eval_package_50\ragas_eval_inputs_with_answers_50.jsonl --run-ragas --judge-backend ollama --judge-model qwen2.5:7b-instruct --embedding-model .\models\bge-m3 --scores .\ragas_eval_package_50\ragas_scores_50.csv --summary .\ragas_eval_package_50\ragas_summary_50.json
```

### 8. 如果你想评估 HTTP 接口行为，再使用 API 模式

只有当你想评估完整服务链路时，再启动 FastAPI：

```powershell
python run.py serve
```

确认你的接口能访问，例如：

```powershell
curl -X POST http://127.0.0.1:8000/query -H "Content-Type: application/json" -d "{\"query\":\"南通办法中深基坑工程如何定义？\"}"
```

返回里必须有 `answer` 和 `contexts`。

确认你的接口能访问，例如：

```powershell
curl -X POST http://127.0.0.1:8000/query -H "Content-Type: application/json" -d "{\"query\":\"南通办法中深基坑工程如何定义？\"}"
```

返回里必须有 `answer` 和 `contexts`。

API 模式示例：

```powershell
python .\ragas_eval_package_50\evaluate_ragas_langchain.py --dataset .\ragas_eval_package_50\ragas_eval_50.jsonl --mode api --url http://127.0.0.1:8000/query --limit 5 --out .\ragas_eval_package_50\ragas_eval_inputs_with_answers_5.jsonl
```

## 调试模式

只想验证脚本和Ragas流程是否能跑，不想调用你的RAG接口，可以用：

```powershell
python evaluate_ragas_langchain.py --dataset ragas_eval_50.jsonl --mode debug_golden_contexts --limit 5 --out debug_inputs.jsonl --run-ragas --judge-backend ollama --judge-model qwen2.5:7b-instruct --embedding-model .\models\bge-m3 --scores debug_scores.csv --summary debug_summary.json
```

注意：这个模式用的是人工 `evidence_contexts`，不能代表你RAG系统真实检索质量。

## 常见报错

### 1. 接口没有返回contexts

说明你的 `/query` 只返回了答案。需要在RAG后端里把检索到的文档chunk一起返回。

推荐后端返回：

```python
return {
    "answer": answer,
    "contexts": [doc.page_content for doc in docs]
}
```

### 2. Qwen thinking相关400错误

如果你用OpenAI兼容Qwen3接口，并出现 `enable_thinking must be set to false`，运行时加：

```powershell
--judge-backend openai_compatible --disable-thinking
```

并确保环境变量设置正确：

```powershell
$env:OPENAI_API_KEY="你的key"
$env:OPENAI_API_BASE="你的base_url"
```

### 3. Ragas版本差异导致metric导入失败

脚本里已经写了常见新旧版本兼容逻辑。如果你本地版本仍报错，先查看版本：

```powershell
python -c "import ragas; print(ragas.__version__)"
```

然后按你版本的Ragas文档调整metrics名称。
