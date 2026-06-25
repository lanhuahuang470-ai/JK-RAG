# 企业 RAG 知识库 — Docker 部署指南

## 目录结构

将本 `rag-ui/` 目录放在你的 `enterprise_rag/` 项目根目录下：

```
enterprise_rag/           ← 你原有的项目
├── config/
├── src/
├── run.py
├── requirements.txt
└── rag-ui/               ← 本目录
    ├── frontend/         ← React 前端
    ├── nginx/            ← Nginx 配置
    ├── Dockerfile.backend
    ├── docker-compose.yml
    ├── .env.example
    └── DEPLOY.md
```

---

## 快速部署（3 步）

### 第 1 步：配置环境变量

```bash
cd rag-ui
cp .env.example .env
# 编辑 .env，填入你的 DASHSCOPE_API_KEY
```

### 第 2 步：构建并启动

```bash
docker compose up -d --build
```

首次构建约需 3–5 分钟（下载依赖）。

### 第 3 步：访问

浏览器打开 **http://localhost:3000**（或你服务器的 IP:3000）

---

## 日常操作

```bash
# 查看运行状态
docker compose ps

# 查看日志
docker compose logs -f backend    # 后端日志
docker compose logs -f frontend   # 前端日志

# 停止服务
docker compose down

# 停止并清除数据（慎用！会删除已上传文档和向量索引）
docker compose down -v

# 更新代码后重新构建
docker compose up -d --build
```

---

## 修改访问端口

编辑 `.env`：
```
PORT=8080
```
然后 `docker compose up -d` 重启即可，通过 `http://your-server:8080` 访问。

---

## 数据持久化

知识库数据存储在 Docker Named Volumes 中，`docker compose down` 不会丢失数据：

| Volume | 内容 |
|--------|------|
| `rag_uploads` | 上传的原始文档 |
| `rag_vectorstore` | FAISS 向量索引 |

如需备份：
```bash
docker run --rm -v rag_vectorstore:/data -v $(pwd):/backup alpine \
  tar czf /backup/vectorstore-backup.tar.gz /data
```

---

## FastAPI 健康检查

后端需要有 `/health` 接口供 Docker 健康检测使用。  
如果你的 `src/api/main.py` 中还没有，添加：

```python
@app.get("/health")
def health():
    return {"status": "ok"}
```

---

## 常见问题

**Q: 前端显示"网络错误"**  
A: 确认后端容器正常运行：`docker compose ps`，查看 `rag-backend` 状态是否为 `healthy`。

**Q: 上传文档失败**  
A: 检查后端日志：`docker compose logs backend`，通常是 DASHSCOPE_API_KEY 未配置或格式错误。

**Q: 想从外网访问**  
A: 确保服务器防火墙开放对应端口（默认 3000），或配置 Nginx 反向代理绑定域名。
