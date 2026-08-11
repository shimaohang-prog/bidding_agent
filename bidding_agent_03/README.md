# 智能招投标问答机器人 Web 版

本目录是在 `bidding_agent_02` 基础上的安全独立副本。保留原六分类 RAG 与 `main.py` CLI，同时新增 FastAPI、Vue 3、MySQL 会话持久化、Redis 上下文/可靠事件重放、WebSocket 流式回答、停止/重连和私有文件证据层。

开始前请阅读：

- [只读审计](docs/audit_report.md)
- [运行与部署](docs/deployment.md)
- [WebSocket 协议](docs/websocket_protocol.md)
- [分层验收报告](verification_report.md)

最小本地启动必须使用单 API Worker；Milvus Lite 不支持多 Worker 高并发共享。
