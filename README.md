# 企业级 RAG 知识库问答系统

基于 **LangChain + FAISS + Qwen3-Embedding + Qwen-Max** 构建的生产就绪 RAG 系统。

---

## 技术栈

| 组件 | 选型 | 说明 |
|------|------|------|
| 向量数据库 | FAISS | 高效本地向量检索，支持增量更新 |
| 嵌入模型 | Qwen3-Embedding（text-embedding-v4） | 通过 DashScope OpenAI 兼容接口调用 |
| 大语言模型 | qwen-max | 通过 DashScope OpenAI 兼容接口调用 |
| 问答链策略 | LangChain Stuff | 将检索上下文全部拼接后一次送入 LLM |
| PDF 处理 | PyPDF2 | 逐页提取文本，处理多页文档 |
| Word 处理 | python-docx | 提取段落 + 表格内容 |
| 文本切分 | RecursiveCharacterTextSplitter | chunk_size=1000, chunk_overlap=200 |
| API 框架 | FastAPI + uvicorn | 异步 REST API |

---

## 项目结构

```
enterprise_rag/
├── config/
│   └── settings.py          # 全局配置（从环境变量加载）
├── src/
│   ├── document_processor/
│   │   └── processor.py     # PDF/DOCX 提取 + 文本切分
│   ├── embeddings/
│   │   └── qwen3_embeddings.py  # Qwen3-Embedding 适配器
│   ├── vectorstore/
│   │   └── faiss_store.py   # FAISS 索引管理
│   ├── llm/
│   │   └── qwen_llm.py      # Qwen LLM LangChain 适配器
│   ├── chain/
│   │   └── rag_chain.py     # RAG 问答链（stuff 策略）
│   ├── api/
│   │   └── main.py          # FastAPI 应用
│   └── utils.py             # 日志工具
├── tests/
│   └── test_rag.py          # 单元测试
├── data/
│   ├── uploads/             # 临时文件存储
│   └── vectorstore/         # FAISS 索引持久化
├── run.py                   # CLI 入口
├── requirements.txt
└── .env.example             # 环境变量模板
```

---

## 快速开始

### 1. 环境配置

```bash
# 克隆/创建项目后安装依赖
pip install -r requirements.txt

# 复制并填写环境变量
cp .env.example .env
# 编辑 .env，填入 DASHSCOPE_API_KEY
```

### 2. 方式一：命令行（快速测试）

```bash
# 索引文档
python run.py index --files your_doc.pdf your_manual.docx

# 提问
python run.py query "公司的退款政策是什么？"

# 全量重建索引
python run.py index --files docs/*.pdf --rebuild
```

### 3. 方式二：HTTP API 服务

```bash
# 启动服务（默认 0.0.0.0:8000）
python run.py serve

# 开发模式（热重载）
python run.py serve --reload
```

**API 文档**：访问 `http://localhost:8000/docs`

#### 上传文档

```bash
curl -X POST "http://localhost:8000/documents/upload" \
  -F "file=@your_document.pdf"
```

#### 问答查询

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "公司的退款政策是什么？", "top_k": 5}'
```

#### 查看索引状态

```bash
curl http://localhost:8000/index/status
```

---

## 核心配置说明

所有配置通过环境变量或 `.env` 文件注入：

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `DASHSCOPE_API_KEY` | **必填** | 阿里云百炼平台 API Key |
| `EMBEDDING_MODEL` | `text-embedding-v4` | 嵌入模型（Qwen3-Embedding） |
| `LLM_MODEL` | `qwen-max` | 大语言模型 |
| `CHUNK_SIZE` | `1000` | 文档切分长度（字符数） |
| `CHUNK_OVERLAP` | `200` | 相邻 chunk 重叠长度 |
| `RETRIEVAL_TOP_K` | `5` | 检索返回 Top-K 数量 |
| `FAISS_INDEX_PATH` | `data/vectorstore/faiss_index` | 索引持久化路径 |

---

## Stuff 问答链原理

```
用户问题
    ↓
[FAISS] 相似度检索 → Top-K 个相关 chunk
    ↓
[Stuff 策略] 将所有 chunk 拼接为单一 context
    ↓
Prompt = "参考资料：{context}\n问题：{question}"
    ↓
[Qwen-Max] 生成最终回答
    ↓
结构化响应（answer + sources）
```

**Stuff vs 其他策略**：
- `stuff`：适合上下文短、Top-K 小的场景，延迟最低
- `map_reduce`：适合大量文档，先分别问每个 chunk 再汇总
- `refine`：迭代精炼，适合需要综合多段落的复杂问题

---

## 运行测试

```bash
pytest tests/ -v
```

---

## 生产部署建议

1. **API Key 安全**：使用 Kubernetes Secrets 或云平台密钥管理服务，不要提交 `.env` 文件
2. **FAISS 持久化**：挂载 PVC 或定期同步索引到对象存储（OSS/S3）
3. **并发优化**：生产环境设置 `API_WORKERS > 1`，注意 FAISS 索引需加读写锁
4. **监控**：接入 Prometheus + Grafana 监控 API 延迟和 token 消耗
5. **增量更新**：文档更新后调用 `/documents/upload` 接口，无需重建全量索引
