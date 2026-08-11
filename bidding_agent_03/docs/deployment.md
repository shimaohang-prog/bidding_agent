# 运行与部署

所有命令都从 `D:\AI\Pythoncode\bidding_agent\bidding_agent_03` 执行。先把 `.env.example` 复制为 `.env`，再填入自己的数据库密码和 API Key；不要提交 `.env`。

## Windows 本地开发

```powershell
cd D:\AI\Pythoncode\bidding_agent\bidding_agent_03
Copy-Item .env.example .env
python -m pip install -r requirements.txt
alembic upgrade head
python -m backend.cli create-user admin
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --workers 1 --reload
```

另开终端运行可靠文件 Worker：

```powershell
cd D:\AI\Pythoncode\bidding_agent\bidding_agent_03
python -m backend.workers.file_worker
```

另开终端运行前端（需要 Node.js 22 和 pnpm）：

```powershell
cd D:\AI\Pythoncode\bidding_agent\bidding_agent_03\frontend
pnpm install
pnpm run dev
```

浏览器访问 `http://localhost:5173`。开发环境若使用 HTTP，`.env` 中必须设置 `COOKIE_SECURE=false`；生产必须恢复 `true`。

## 单 Worker Milvus Lite MVP

```powershell
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 1
```

必须保持 `MILVUS_MODE=lite`、`API_WORKERS=1`。Lite 数据库是本地文件，不能作为多 Worker 高并发共享存储；每个 API 进程也会重复加载 BGE/Torch。文件 Worker 与 API 对私有 Lite 库的并发仍需谨慎安排，遇到文件锁应停止占用者，不能删除数据库。

产品 CSV 修改后仍显式执行：

```powershell
python -m builders.build_vectors --categories product --rebuild
```

这不会自动触发，执行前必须停止使用 product 库的进程。

## Docker Compose

准备 TLS 证书为 `deploy/certs/fullchain.pem` 与 `deploy/certs/privkey.pem`，并设置 Compose 密码变量：

```powershell
$env:MYSQL_PASSWORD='replace-with-strong-value'
$env:MYSQL_ROOT_PASSWORD='replace-with-another-strong-value'
docker compose up -d mysql redis
docker compose run --rm api alembic upgrade head
docker compose run --rm api python -m backend.cli create-user admin
docker compose up -d api file-worker nginx
```

默认仍是单 Worker Milvus Lite MVP。`docker compose --profile milvus-server up -d` 提供 Milvus Standalone、etcd 和 MinIO 基础设施；当前五个公共分类仍是既有 Lite 快照。在完成五类数据迁移、服务 URI 适配、功能回归和压测前，不得提高 API Worker 数。生产镜像版本应纳入发布时的安全更新流程。

## 迁移、构建和测试

```powershell
# 已初始化的 Alembic；模型变化时生成迁移
alembic revision --autogenerate -m "describe change"
alembic upgrade head

# 前端
cd frontend
pnpm run typecheck
pnpm run build
cd ..

# 全部单元与隔离集成测试
python -m pytest -q

# 只运行原有 RAG 回归
python -m pytest -q tests/test_hybrid_retrieval.py tests/test_models_and_fusion.py tests/test_six_category_architecture.py
```

## 压测和观测

先通过浏览器登录并在环境变量放置短期 Cookie，不要把 Cookie 写进命令历史或代码：

```powershell
$env:BIDDING_COOKIE='access_token=short-lived-value'
python tests/load/ws_benchmark.py --url wss://localhost/api/v1/ws/chat --origin https://localhost --conversation-id YOUR_UUID --concurrency 10 --requests 100
```

脚本输出并发数、P50/P95/P99、首 Token 时间、总响应时间和错误率。Lite 阶段只压测单 Worker。CPU/内存请同时记录 Windows 性能监视器或 `docker stats`；外部 API 的 429 需单独计入限流指标。

## 原 CLI

```powershell
python main.py "安徽省某招标项目适用哪些法律政策？" --show-plan
```

CLI 继续复用原同步领域 Service；它会使用真实 DeepSeek/Tavily 配置并可能读取本地 Milvus，不能把未执行的命令视为已验证。
