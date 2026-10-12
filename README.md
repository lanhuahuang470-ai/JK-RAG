# 企业级 RAG 知识库问答系统

基于 **LangChain + FAISS + Qwen3-Embedding + Qwen-Max** 构建的生产就绪 RAG 系统，支持**混合检索（向量 + BM25）+ Rerank 精排**、多轮对话与查询改写。

---

## 技术栈

| 组件 | 选型 | 说明 |
|------|------|------|
| 向量数据库 | FAISS | 高效本地向量检索，支持增量更新 |
| 嵌入模型 | Qwen3-Embedding（text-embedding-v4） | 通过 DashScope OpenAI 兼容接口调用 |
| 大语言模型 | 由 `LLM_MODEL` 决定（默认 `qwen-max`） | 通过 DashScope OpenAI 兼容接口调用 |
| 文档切分 | 层次切分（Hierarchical Chunking） | 按章/节/条结构切分，无结构自动回退递归切分 |
| 混合检索 | BM25 + 向量双路召回，RRF 融合 | 补齐条款编号等精确匹配短板 |
| 精排策略 | DashScope gte-rerank-v2 | cross-encoder 对召回候选二次排序 |
| 问答链策略 | LangChain Stuff | 将检索上下文全部拼接后一次送入 LLM |
| 多轮对话 | 会话管理 + 查询改写 | 指代/省略自动补全，历史会话持久化（内存） |
| 评估体系 | RAGAS（50 题评测集） | 四维量化：忠实度 / 相关性 / 精确率 / 召回率 |
| API 框架 | FastAPI + uvicorn | 异步 REST API |

---

## 项目结构

```
JK-RAG/
├── config/
│   └── settings.py          # 全局配置（从环境变量加载）
├── src/
│   ├── llm.py               # QwenLLM：大语言模型封装（DashScope 兼容接口）
│   ├── rag/                 # RAG 核心（文档处理 + 嵌入 + 向量库 + 问答链）
│   │   ├── document_processor.py  # 层次切分文档处理器
│   │   ├── embeddings.py    # DashScope 嵌入模型
│   │   ├── vectorstore.py   # FAISS 向量库管理
│   │   ├── hybrid_retriever.py  # 混合检索（BM25 + 向量 RRF 融合）
│   │   ├── reranker.py     # DashScope Rerank 精排器（gte-rerank-v2）
│   │   ├── rag_chain.py     # RAG 问答链（三段式检索 + Stuff 生成）
│   │   └── query_rewriter.py # 查询改写（单轮优化 + 多轮指代消解）
│   ├── api/                 # API 层
│   │   ├── main.py          # FastAPI 应用
│   │   └── conversation.py  # 会话管理（多轮对话）
│   └── utils.py             # 通用工具（日志等）
├── ragas_eval_package_50/   # RAGAS 评估包（50 题评测集）
├── rag-ui/                  # 前端 UI（React + Vite）
│   └── frontend/
├── data/
│   ├── documents/           # 原始文档（PDF/DOCX）
│   ├── uploads/             # 上传临时文件
│   └── vectorstore/         # FAISS 索引持久化
├── .venv/                   # 主虚拟环境（RAG 运行时）
├── .venv-ragas/             # 评估专用虚拟环境（可选）
├── run.py                   # CLI 入口
├── requirements.txt
├── .env.example             # 环境变量模板
└── README.md
```

---

## 架构设计

### 分层架构

```
┌─────────────────────────────────────────┐
│              API / CLI 层               │
│     (FastAPI / run.py / 会话管理)        │
├─────────────────────────────────────────┤
│              RAG 问答链                 │
│  查询改写 → FAISS 检索 → Stuff 生成     │
├─────────────────────────────────────────┤
│              RAG 核心层                 │
│  文档处理 / 嵌入 / 向量库 / LLM 封装    │
└─────────────────────────────────────────┘
```

### 问答链路

```
用户问题（可携带会话历史）
   ↓
查询改写：单轮检索优化 / 多轮指代消解（有历史时强制启用）
   ↓
┌──── 三段式检索 ────────────────────────────────┐
│ ① 召回：BM25 关键词 + 向量语义双路，RRF 融合    │
│ ② 精排：gte-rerank cross-encoder 重排 Top-K     │
└──────────────────────────────────────────────────┘
   ↓
Stuff 策略：上下文 + 历史拼接，一次送入 LLM
   ↓
答案 + 条款级来源（文件名 · 章 > 节 > 条）+ 检索置信度
```

---

## 快速开始

### 0. 前置准备

```bash
# 1. 创建并激活虚拟环境
python -m venv .venv
.\.venv\Scripts\activate

# 2. 安装后端依赖
pip install -r requirements.txt

# 3. 配置环境变量
cp .env.example .env
# 编辑 .env，填入 DASHSCOPE_API_KEY
```

### 1. 构建知识库（第一次必做）

把你的 PDF / Word 文档放到 `data/documents/` 目录下，然后建索引：

```bash
# 构建知识库索引（以后新增文档用增量，不加 --rebuild）
python run.py index --dir data/documents --rebuild
```

### 2. 方式一：网页版（推荐）

需要同时启动**后端 API** 和 **前端页面**，开两个终端：

**终端 1 — 启动后端：**
```bash
.\.venv\Scripts\activate
python run.py serve
# 后端运行在 http://localhost:8000
# API 文档：http://localhost:8000/docs
```

**终端 2 — 启动前端：**
```bash
cd rag-ui/frontend
npm install     # 第一次需要安装依赖
npm run dev
# 前端运行在 http://localhost:5173
```

打开浏览器访问 **http://localhost:5173** 即可使用网页版问答。

**网页版功能：**
- 左侧历史对话列表，支持新建/切换/删除对话
- 多轮对话自动携带上下文，支持指代理解（如"它是什么"、"再详细说说"）
- 每条回答展示条款级来源（文件名 · 章 > 节 > 条）和检索置信度

> 前端通过 Vite 代理将 `/api/*` 请求转发到后端 `http://localhost:8000`，无需额外配置跨域。

### 3. 方式二：命令行（快速测试）

```bash
# 提问
python run.py query "南通深基坑工程如何定义？"

# 启用查询改写（检索前用 LLM 优化问题）
python run.py query "基坑挖多深算深的？" --rewrite
```

### 4. 方式三：纯 API 调用

```bash
# 启动服务
python run.py serve

# 基础问答（无会话）
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "南通深基坑工程如何定义？", "top_k": 5}'

# 上传文档
curl -X POST "http://localhost:8000/documents/upload" \
  -F "file=@your_document.pdf"

# 查看索引状态
curl http://localhost:8000/index/status
```

#### 多轮对话 API

```bash
# 创建新会话
curl -X POST "http://localhost:8000/conversations"

# 获取会话列表
curl "http://localhost:8000/conversations"

# 获取会话详情（含历史消息）
curl "http://localhost:8000/conversations/{会话ID}"

# 在指定会话中发送消息（自动携带历史上下文）
curl -X POST "http://localhost:8000/conversations/{会话ID}/messages" \
  -H "Content-Type: application/json" \
  -d '{"question": "它的适用范围是什么？", "top_k": 5}'

# 删除会话
curl -X DELETE "http://localhost:8000/conversations/{会话ID}"
```

> 多轮对话会自动用 LLM 改写当前问题，结合历史上下文理解指代和省略。

---

## 核心配置说明

所有配置通过环境变量或 `.env` 文件注入：

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `DASHSCOPE_API_KEY` | **必填** | 阿里云百炼平台 API Key |
| `EMBEDDING_MODEL` | `text-embedding-v4` | 嵌入模型（Qwen3-Embedding） |
| `LLM_MODEL` | `qwen-max` | 大语言模型 |
| `LLM_TEMPERATURE` | `0.1` | 生成温度 |
| `CHUNK_SIZE` | `1000` | 文档切分长度（字符数） |
| `CHUNK_OVERLAP` | `200` | 相邻 chunk 重叠长度 |
| `RETRIEVAL_TOP_K` | `5` | 检索返回 Top-K 数量 |
| `RETRIEVAL_SCORE_THRESHOLD` | `0.5` | 检索分数阈值 |
| `FAISS_INDEX_PATH` | `data/vectorstore/faiss_index` | 索引持久化路径 |
| `ENABLE_QUERY_REWRITE` | `false` | 是否启用查询改写（多轮对话时强制启用） |
| `ENABLE_HYBRID_RETRIEVAL` | `true` | 混合检索（向量 + BM25 RRF 融合） |
| `RETRIEVAL_FETCH_K` | `20` | 召回阶段候选数量（融合 / Rerank 前） |
| `ENABLE_RERANK` | `true` | Rerank 精排（cross-encoder 二次排序） |
| `RERANK_MODEL` | `gte-rerank-v2` | DashScope Rerank 模型名 |
| `API_HOST` | `0.0.0.0` | API 监听地址 |
| `API_PORT` | `8000` | API 监听端口 |
| `LOG_LEVEL` | `INFO` | 日志级别 |

---

## 文档切分策略

默认使用**层次切分（Hierarchical Chunking）**，按文档的章/节/条结构切分：

- **有层次结构的文档**（如规范、规章）：按章/节/条边界切分，每个 chunk 对应完整条款
- **无层次结构的文档**：自动回退到递归字符切分
- **超长条款**：自动回退到字符级切分，保证 chunk 大小可控
- **元数据丰富**：每个 chunk 携带层次路径（如"第一章 总则 > 第三条"）

---

## 评估（RAGAS）

项目包含 RAGAS 评估工具包，基于 50 题领域评测集从四个维度量化系统质量。

### 评估环境

评估使用独立的虚拟环境 `.venv-ragas/`，避免与主环境依赖冲突。

### 基线结果（纯向量检索）

| 指标 | 分数 | 说明 |
|------|------|------|
| Faithfulness（忠实度） | 0.765 | 回答是否忠于检索上下文 |
| Answer Relevancy（相关性） | 0.806 | 回答是否切题 |
| Context Precision（精确率） | 0.708 | 检索结果中相关上下文占比 |
| Context Recall（召回率） | 0.767 | 关键依据是否被完整召回 |

> Judge 模型：DeepSeek（deepseek-v4-flash）｜评估嵌入：bge-m3

### 运行评估

```bash
# 激活评估环境
.\.venv-ragas\Scripts\activate

# 运行评估（详见 ragas_eval_package_50/README_Ragas运行说明.md）
python ragas_eval_package_50/evaluate_ragas_langchain.py \
  --dataset ragas_eval_package_50/ragas_eval_50.jsonl \
  --mode direct \
  --run-ragas \
  --judge-backend openai_compatible \
  --judge-model deepseek-v4-flash \
  --embedding-model dashscope
```

### 评估指标

- **Faithfulness**（忠实度）：回答是否基于上下文，量化幻觉风险
- **Answer Relevancy**（回答相关性）：回答是否切题
- **Context Precision**（上下文精确率）：检索结果中相关上下文占比
- **Context Recall**（上下文召回率）：关键依据是否被完整召回

---

## 生产部署建议

1. **API Key 安全**：使用 Kubernetes Secrets 或云平台密钥管理服务，不要提交 `.env` 文件
2. **FAISS 持久化**：挂载 PVC 或定期同步索引到对象存储（OSS/S3）
3. **并发优化**：生产环境设置 `API_WORKERS > 1`，注意 FAISS 索引需加读写锁
4. **监控**：接入 Prometheus + Grafana 监控 API 延迟和 token 消耗
5. **增量更新**：文档更新后调用 `/documents/upload` 接口，无需重建全量索引

---

## 常见问题

### 两个虚拟环境有什么区别？

- **`.venv/`**：主运行环境，包含 RAG 系统核心依赖（~217MB，66 个包）
- **`.venv-ragas/`**：评估专用环境，包含 RAGAS、PyTorch、Transformers 等重型依赖（~1.2GB，108 个包）

主环境足够日常运行使用，评估环境仅在需要运行 RAGAS 评估时才需要。

### 层次切分是什么意思？

层次切分是按文档的自然结构（章/节/条）来切分文档，而不是简单按字符数切。这样每个 chunk 更可能对应一个完整的语义单元（如一条完整的规定），检索效果更好。如果文档没有明显的层次结构，会自动回退到普通的递归字符切分。
