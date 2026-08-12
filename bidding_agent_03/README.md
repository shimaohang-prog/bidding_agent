# 智能招投标问答机器人 Web 版

本目录是在 `bidding_agent_02` 基础上的安全独立副本。保留原六分类 RAG 与 `main.py` CLI，同时新增 FastAPI、Vue 3、MySQL 会话持久化、Redis 上下文/可靠事件重放、WebSocket 流式回答、停止/重连和私有文件证据层。网页端按会话独立保存生成状态，同一用户可在多个会话中并发提问；CLI 支持单问题流式输出和多问题异步并发。

开始前请阅读：

- [只读审计](docs/audit_report.md)
- [运行与部署](docs/deployment.md)
- [WebSocket 协议](docs/websocket_protocol.md)
- [分层验收报告](verification_report.md)

最小本地启动必须使用单 API Worker；单进程内可处理多个异步问答任务，但 Milvus Lite 不适合作为无限并发或多 Worker 共享存储。正式扩容需迁移 Milvus Server。

网页后端运行时会独占本地 Milvus Lite 数据目录。此时执行 `python main.py`，CLI 会检测 `127.0.0.1:8000`，提示输入 Web 用户名和密码，并通过同一个 WebSocket 后端并发提交问题；不会再由第二个 Python 进程直接打开 Lite 数据库。使用 `--local` 可强制本地模式，但必须先停止网页后端和其他占库进程。
