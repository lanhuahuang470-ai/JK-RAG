# [姓名]

**求职方向：AI Agent 应用开发 / RAG 工程师 / Python 后端开发**

[手机号] · [邮箱] · [所在城市] · [GitHub / Gitee 链接]

## 个人优势

- 具备 AI Agent 应用从工具封装、决策路由、知识库检索、网络搜索兜底到 API / 前端交付的完整项目实践。
- 熟悉 LangChain、FAISS、Qwen / DashScope、FastAPI，能够独立搭建“RAG 优先 + Agent 判断 + 外部工具兜底”的智能问答系统。
- 具备 Agent 工程化意识，实践过 RAGTool、WebSearchTool、置信度阈值路由、LLM 可靠性判定、查询改写和结构化响应返回。
- 具备 Python 后端与 React 前端协作开发能力，能够使用 Docker Compose、Nginx 完成应用容器化部署。

## 技术能力

- **AI Agent / 大模型应用：** Agent 路由决策、工具调用、RAG、Prompt Engineering、LangChain、Qwen-Max、Qwen3-Embedding、Query Rewriting
- **检索与文档处理：** FAISS、向量检索、Top-K 检索、PDF / DOCX 解析、递归切分、语义切分、层级切分、来源追踪
- **后端开发：** Python、FastAPI、Pydantic、RESTful API、异步文件上传、环境变量配置、日志与异常处理
- **前端开发：** React、Vite、Fetch API、文件拖拽上传、对话式交互、检索来源与路由结果展示
- **工程化部署：** Docker、Docker Compose、Nginx、健康检查、数据卷持久化、CLI 工具

## 项目经历

### 企业级 RAG + Agent 智能知识库问答系统 ｜ 核心开发者

**项目时间：** [20XX.XX – 20XX.XX]  
**技术栈：** Python、LangChain、FAISS、Qwen-Max、Qwen3-Embedding、DashScope、FastAPI、React、Vite、Docker Compose、Nginx

**项目简介：**  
面向企业制度、工程规范等专业文档场景，设计并实现一套“本地知识库优先、Agent 动态决策、网络搜索兜底”的智能问答系统。系统支持 PDF / DOCX 文档上传、向量索引增量更新、自然语言问答、答案来源追踪、检索置信度评估，以及在本地知识不足时自动调用外部搜索工具补充回答。

**核心工作：**

- 设计并实现 RAG + Agent 问答架构，将知识库检索能力封装为 `RAGTool`，将外部搜索能力封装为 `WebSearchTool`，由 `FallbackAgent` 统一完成工具选择、路由判断和结果返回。
- 实现基于置信度的 Agent 决策策略：高置信度问题直接采用本地 RAG 回答，低置信度问题自动触发 Bocha / Tavily 网络搜索兜底，临界置信度问题交由 LLM 判断 RAG 答案是否可靠后再决定是否切换工具。
- 搭建完整 RAG 链路：解析 PDF / DOCX 文档，使用 Qwen3-Embedding 生成向量并写入 FAISS，通过 LangChain Stuff Chain 将 Top-K 检索片段注入 Qwen-Max，生成基于知识库内容的回答。
- 封装 FAISS 索引管理模块，实现索引创建、磁盘持久化、启动热加载、增量写入、相似度检索、分数返回和索引重置，支持文档更新后持续扩充知识库。
- 设计递归、语义和层级三种可配置文档切分策略；针对规章制度、技术规范等结构化文档识别“章—节—条”等层级，并在元数据中保留层级路径，提升检索片段的语义完整性和来源可解释性。
- 引入 LLM 查询改写，将口语化、模糊问题转换为更适合专业文档检索的查询表达；同时配置异常回退与长度校验，改写失败时自动使用原始问题，保证主链路可用性。
- 在回答结果中返回答案、来源文档、来源分数、检索置信度、工具路由路径和接口耗时，使 Agent 的决策过程对前端和调试人员可观察。
- 使用 FastAPI 提供文档上传、Agent 问答、索引状态、健康检查和索引重置等 REST API，并通过 Pydantic 完成请求约束和结构化响应。
- 使用 React + Vite 开发对话式 Web 界面，支持拖拽上传文档、动态选择 Top-K、展示知识来源、检索分数和 Agent 路由结果，便于用户判断回答依据。
- 编写 CLI 入口，支持批量索引、增量更新、全量重建、单次查询和 API 服务启动，兼顾本地调试和服务化运行。
- 使用 Docker Compose 编排前后端服务，以 Nginx 反向代理 FastAPI；配置容器健康检查、服务依赖和索引目录持久化，形成可复用的一键部署方案。

## 教育经历

### [学校名称] · [专业名称] · [学历]

[20XX.09 – 20XX.06]

- 主修课程：[数据结构、计算机网络、操作系统、数据库原理、机器学习等，请按实际情况填写]
- 成绩 / 排名：[如无明显优势可删除]

## 实习 / 工作经历

### [公司名称] · [岗位名称]

[20XX.XX – 20XX.XX]

- [使用“动作 + 技术 / 方法 + 结果”的方式填写，尽量包含可验证数据]
- [若暂无相关经历，可删除本节，将项目经历放在教育经历之前]

## 其他信息

- GitHub / Gitee：[项目仓库链接]
- 技术博客：[链接；没有可删除]
- 英语水平：[CET-4 / CET-6 / 其他；按实际填写]
- 获奖 / 证书：[按实际填写；没有可删除]

---

## 招聘平台精简版项目描述

独立开发企业级 RAG + Agent 智能知识库问答系统，基于 LangChain、FAISS、Qwen-Max 和 Qwen3-Embedding 实现 PDF / DOCX 文档解析、向量索引增量更新、Top-K 检索、答案生成与来源追踪。项目核心不是单纯 RAG，而是在 RAG 之上设计了 `FallbackAgent` 决策层：将本地知识库封装为 `RAGTool`，将 Bocha / Tavily 网络搜索封装为 `WebSearchTool`，通过检索置信度阈值和 LLM 可靠性判定实现“本地知识库优先、低置信度自动搜索兜底、临界问题智能判断”的工具路由机制。后端使用 FastAPI 提供 Agent 问答、文档上传、索引状态和健康检查接口，前端使用 React 展示对话、来源、分数和路由结果，并通过 Docker Compose + Nginx 完成容器化部署。

## 面试时可重点展开

1. 为什么这个项目是 Agent，而不是普通 RAG：它具备工具封装、状态判断、路由决策和兜底工具调用。
2. `FallbackAgent` 的决策逻辑：高置信度走 RAG，低置信度走 Web Search，临界区由 LLM 判断可靠性。
3. RAGTool 和 WebSearchTool 如何解耦，为什么这样设计更方便扩展更多工具。
4. 检索置信度如何参与 Agent 路由，以及如何把 route、confidence、sources 暴露给前端。
5. 查询改写、层级切分和来源追踪分别解决了 RAG / Agent 系统中的哪些真实问题。
