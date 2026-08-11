# bidding_agent_02 只读审计报告

审计时间：2026-08-06。审计对象为 `bidding_agent_02`；整个审计未读取 `.env` 内容，未连接、删除、覆盖或重建任何 Milvus 数据。

## 实际目录与入口

入口是 `main.py`，领域调用链是 `create_retrieval_plan` → `execute_retrieval_plan` → `build_answer_context` → `generate_answer`。代码目录为 `builders`、`common`、`planning`、`retrieval`、`ranking`、`generation`、`tests`；审计时不存在 FastAPI、Vue、SQLAlchemy、Redis 或任务 Worker。

## 模块依赖和同步阻塞点

- Planner 和 Reranker 使用同步 DeepSeek Function Calling。
- Answer Generator 使用同步、非流式 DeepSeek completion。
- Tavily 使用 `requests.Session`；DeepSeek 使用 `requests.post` 和 `time.sleep` 重试。
- BGE 模型加载和编码、PyMilvus 建连、Dense/BM25 两路搜索都是同步调用。
- Web API 不能直接调用上述同步路径；新副本通过 AnyIO 有界线程桥接，并为 DeepSeek Answer、Tavily 新增共享 `httpx.AsyncClient` 异步实现。

## 数据库现状

| 分类 | 模式 | 记录数 | 索引 |
|---|---|---:|---|
| enterprise | 独立 Milvus Lite | 39,037 | Dense COSINE + Sparse BM25 |
| tender | 独立 Milvus Lite | 8,136 | Dense COSINE + Sparse BM25 |
| product | 独立 Milvus Lite | 24,408 | Dense COSINE + Sparse BM25 |
| laws | 独立 Milvus Lite | 2,094 | Dense COSINE + Sparse BM25 |
| policy | 独立 Milvus Lite | 579 | Dense COSINE + Sparse BM25 |
| news | web-only | 不适用 | 无本地向量库 |

CSV 共约 71 MB；法律 49 个文件，政策 93 个文件。源项目测试基线为 23 项通过，退出码 0。

## 实施计划与安全边界

1. 复制到 `bidding_agent_03`，排除 `.env` 和缓存目录；数据、五个向量库复制成独立副本。
2. 新增 MySQL Web 元数据、Redis、鉴权和 REST，禁止恢复业务 MySQL 查询。
3. 复用现有 Planner、过滤白名单、Dense/BM25、两级 RRF、去重和 Reranker。
4. 新增流式 DeepSeek、可靠 WebSocket 事件、取消与重放。
5. 新增 Vue、私有文件 Worker、Nginx 和部署文档。
6. 静态/单元、API 集成、外部服务、本地 Milvus、压测分开报告。
