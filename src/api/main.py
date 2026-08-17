# FastAPI应用入口 + 生命周期管理
import os
import time
from contextlib import asynccontextmanager
from collections import defaultdict
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from src.api.routes import router

# 限流配置
RATE_LIMIT_WINDOW = 60    # 时间窗口（秒）
RATE_LIMIT_MAX = 200      # 每窗口最大请求数
_rate_limit_store: dict = defaultdict(list)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """基于IP的滑动窗口限流"""
    async def dispatch(self, request: Request, call_next):
        # 跳过静态文件和文档页面
        if request.url.path.startswith("/api/static") or request.url.path in ("/docs", "/openapi.json", "/redoc"):
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        window_start = now - RATE_LIMIT_WINDOW

        # 清理过期记录
        _rate_limit_store[client_ip] = [t for t in _rate_limit_store[client_ip] if t > window_start]

        if len(_rate_limit_store[client_ip]) >= RATE_LIMIT_MAX:
            return JSONResponse(
                {"detail": f"请求过于频繁，请 {RATE_LIMIT_WINDOW}s 后重试"},
                status_code=429,
            )

        _rate_limit_store[client_ip].append(now)
        return await call_next(request)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """注入安全响应头 + 操作审计日志"""
    async def dispatch(self, request: Request, call_next):
        # 审计日志（跳过静态资源和文档）
        audit_path = request.url.path
        if not audit_path.startswith("/api/static") and audit_path not in ("/docs", "/openapi.json", "/redoc", "/"):
            try:
                db = request.app.state.db_manager
                token = request.headers.get("Authorization", "").removeprefix("Bearer ")
                from src.api.auth import verify_token
                payload = verify_token(token) if token else None
                db.log_event(
                    run_id="api",
                    event_type=request.method,
                    doc_id=payload.get("sub", "anonymous") if payload else "anonymous",
                    details={
                        "path": audit_path,
                        "method": request.method,
                        "client": request.client.host if request.client else "unknown",
                    },
                )
            except Exception:
                pass  # 审计失败不影响业务

        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; "
            "connect-src 'self'; "
            "font-src 'self'"
        )
        return response
from src.storage.database_manager import DatabaseManager
from src.retrieval.search_engine import HybridSearchEngine
from src.agent.agent import DocumentAgent
from src.agent.tools import init_tools

# 全局实例（通过app.state访问）
db_manager: DatabaseManager = None       # type: ignore
search_engine: HybridSearchEngine = None # type: ignore
agent: DocumentAgent = None               # type: ignore


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时加载模型，关闭时释放资源"""
    global db_manager, search_engine, agent

    # 启动
    print("[startup] 加载模型与数据库...")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    db_manager = DatabaseManager()
    db_manager.ensure_admin()
    search_engine = HybridSearchEngine(db_manager, enable_rerank=True)
    init_tools(db_manager, search_engine)
    agent = DocumentAgent(backend="local")

    # 注入到app.state，routes通过request.app.state访问
    app.state.db_manager = db_manager
    app.state.search_engine = search_engine
    app.state.agent = agent

    stats = db_manager.get_stats()
    print(f"[startup] 就绪: {stats['documents']}文档 {stats['chunks']}Chunk {stats['qa_pairs']}QA")
    print(f"[startup] API 服务已启动")

    yield  # ← 服务运行中

    # 关闭
    print("[shutdown] 释放资源...")
    if db_manager:
        db_manager.close()


app = FastAPI(
    title="企业文档智能管理系统",
    description="基于 Agent 的智能文档库与知识管理 API",
    version="1.0.0",
    lifespan=lifespan,
)

# 安全头
app.add_middleware(SecurityHeadersMiddleware)

# 限流
app.add_middleware(RateLimitMiddleware)

# CORS跨域（生产环境应改为具体域名白名单）
ALLOWED_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS if o.strip()],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/api/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

app.include_router(router, prefix="/api/v1")


@app.get("/")
def root():
    from fastapi.responses import FileResponse
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "前端未构建", "docs": "/docs"}
