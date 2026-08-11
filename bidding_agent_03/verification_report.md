# verification_report

验收日期：2026-08-06。工作目录：`D:\AI\Pythoncode\bidding_agent\bidding_agent_03`。

## 已完成结果

- `bidding_agent_03` 是独立副本，目标中不存在 `.env`。
- 保留五类独立本地向量库、news web-only、Function Calling Planner、元数据白名单、Dense+BM25、分类内/跨任务 RRF、去重、Reranker 降级和 Tavily hostname 二次过滤。
- 新增 FastAPI REST、HttpOnly Cookie 鉴权、SQLAlchemy AsyncSession、六表模型、Alembic、Redis 上下文、Redis Stream 重放、WebSocket 流、取消、幂等、重连宽限期和应用层心跳。
- 新增 Vue 3 + TypeScript 前端、安全 Markdown、代码高亮、引用、上传、停止和重连提示。
- 新增安全上传、SHA-256、随机存储名、路径边界检查、Redis Stream 文件任务 Worker、pending 自动接管和私有 Milvus 权限过滤。
- 新增 Nginx、Docker Compose、部署文档和 WebSocket 协议文档。

## 静态与单元测试

| 项目 | 命令 | 退出码 | 结果 |
|---|---|---:|---|
| 源项目基线 | `python -m pytest -q`（在 bidding_agent_02） | 0 | 23 passed in 4.20s |
| Python 编译 | `python -m compileall -q backend common planning retrieval ranking generation builders` | 0 | 通过 |
| 应用导入 | `python -c "from backend.app import app"` | 0 | 通过；11 个 REST path，另有 WebSocket path |
| 全量隔离测试 | `python -m pytest -q` | 0 | 41 passed in 6.21s；1 个 TestClient 依赖弃用警告 |
| Alembic 离线 SQL | `alembic upgrade head --sql` | 0 | MySQL 迁移 SQL 成功生成 |
| 前端类型检查 | `node node_modules/vue-tsc/bin/vue-tsc.js -b` | 0 | 通过 |
| 前端生产构建 | `node node_modules/vite/bin/vite.js build` | 0 | 103 modules；JS 269.16 kB，gzip 107.67 kB |
| CLI 入口 | `D:\AI\Anaconda3\envs\cpu_default\python.exe main.py --help` | 0 | 参数与 `--show-plan` 正常 |

新增测试覆盖：REST 登录与会话 CRUD、IDOR 隔离、WebSocket 正常序列、ping、stop、resume、幂等、Origin 拒绝、Redis 上下文/seq/重放/取消/并发限制、SQLite 隔离 Repository、DeepSeek Token/[DONE]/429/5xx/超时/无效 JSON/空流、Worker 切块、同步检索线程桥接，以及相同 semantic query 单次编码和分类并发。

## API 隔离集成测试

已使用临时 SQLite 和内存 Cache 替身执行 REST/WebSocket 集成测试；没有连接真实 MySQL 或 Redis。测试产生的资源仅位于 pytest 临时目录并由测试框架清理。

## 真实本地 Milvus 数据测试

基础 Python 首次执行失败，真实错误是缺少 `sentence-transformers`；随后使用依赖齐全的 `cpu_default` Conda 环境执行：

```powershell
$env:RERANK_MODE='rrf'
D:\AI\Anaconda3\envs\cpu_default\python.exe <只读 laws 检索脚本>
```

退出码 0。本地离线加载 `BAAI/bge-base-zh-v1.5`，对 `bidding_agent_03\milvus_db\laws` 返回 3 条 BM25 候选、0 条警告。该测试没有调用 DeepSeek/Tavily，也没有写入或重建向量库。

## 数据安全验证

- 对 `bidding_agent_02` 与 `bidding_agent_03` 的 `data`、`milvus_db` 全部普通文件（排除运行锁文件）执行 SHA-256 对比：`MISMATCH_COUNT=0`。
- `bidding_agent_03\.env`：不存在。
- 新增代码、前端源码、部署和文档中的常见 DeepSeek/Tavily/Bearer 密钥模式扫描：0 命中。
- 没有运行任何 `builders.build_vectors --rebuild`，没有删除或覆盖源项目数据库。

## 未验证项

- 真实 MySQL 8、真实 Redis 的网络集成和 Alembic 在线升级：未验证（本机未提供服务）。
- 真实 DeepSeek 流、Planner、Reranker 和真实 Tavily：未验证（副本没有 `.env`，不读取/复制源密钥）。
- 私有文件真实 PDF/DOCX Embedding 与 Milvus 索引 Worker 端到端：未验证；纯解析/切块、队列协议和权限过滤代码已做单元验证。
- Docker Compose、Nginx TLS/WSS 和 Milvus Server profile 启动：未验证；本机没有 Docker 命令。
- 完整 `main.py` 问答：未验证；需要真实 DeepSeek Key。CLI 入口及原 23 项领域回归已验证。
- 并发压测：未执行，因此 P50/P95/P99、首 Token、CPU、内存、错误率和外部限流均未验证。已提供 `tests/load/ws_benchmark.py`。

## 已知限制

1. Milvus Lite 必须保持 `--workers 1`；文件 Worker 与 API 同时访问私有 Lite 库仍可能遇到 Windows 文件锁。
2. Compose 的 Milvus Server profile 只提供基础设施。五个公共分类仍是 Lite 快照，完成数据迁移、URI 适配与压测前不得启用多 API Worker。
3. Redis Stream 是短期重放，超过 TTL 后客户端需回读 MySQL 历史。
4. 当前前端代码高亮只按需注册 HTML/CSS/JS/TS/JSON/Python/SQL/Bash，以控制包体积。
5. FastAPI 0.141 的 TestClient 发出一项 httpx 迁移弃用警告，不影响当前测试结果，后续应跟随 Starlette 官方测试客户端迁移。

## 下一步

在隔离测试环境启动 MySQL/Redis，执行在线迁移和 API 集成；配置测试用 DeepSeek/Tavily 凭证进行外部服务验证；用临时私有 Milvus 库跑上传 Worker 端到端；最后在单 Worker Lite 上压测。需要多 Worker 时，先完成五类公共向量数据到 Milvus Server 的迁移和功能回归。
