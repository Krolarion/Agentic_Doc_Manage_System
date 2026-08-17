# API路由定义
import os
import uuid
import tempfile
import traceback
from pathlib import Path
from typing import List
from src.ingestion.document_parser import DocumentParser
from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Request, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.api.auth import (
    create_token, verify_password, hash_password,
    get_current_user, require_admin
)

router = APIRouter()
STATIC_DIR = Path(__file__).parent / "static"


# 请求/响应模型

class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)

class SearchRequest(BaseModel):
    query: str = Field(..., description="搜索查询", min_length=1)
    top_k: int = Field(default=5, ge=1, le=20, description="返回结果数")

class ChatRequest(BaseModel):
    message: str = Field(..., description="用户消息", min_length=1)
    reset: bool = Field(default=False, description="是否重置对话历史")


# 依赖注入：通过app.state获取全局实例

def get_db(request: Request):
    return request.app.state.db_manager

def get_engine(request: Request):
    return request.app.state.search_engine

def get_agent(request: Request):
    return request.app.state.agent


# 前端页面

@router.get("/app", tags=["Frontend"])
@router.get("/app/{path:path}", tags=["Frontend"])
def serve_frontend(path: str = ""):
    """Serve SPA frontend"""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "前端文件不存在，请先构建前端"}


# 认证端点（无需登录）

@router.post("/auth/login", tags=["Auth"])
def login(req: LoginRequest, request: Request):
    """用户登录，返回JWT token"""
    db = request.app.state.db_manager
    user = db.get_user(req.username)
    if not user or not verify_password(req.password, user["password_hash"]):
        raise HTTPException(401, "用户名或密码错误")

    token = create_token(user["username"], user["user_id"], user["role"])
    return {
        "token": token,
        "user": {"username": user["username"], "role": user["role"], "user_id": user["user_id"]}
    }


@router.get("/auth/me", tags=["Auth"])
def current_user(user=Depends(get_current_user)):
    """获取当前登录用户信息"""
    return user


# 需要登录的端点

@router.post("/documents/upload", tags=["Documents"])
async def upload_document(
    file: UploadFile = File(...),
    request: Request = None,  # type: ignore
    db=Depends(get_db),
    engine=Depends(get_engine),
    user=Depends(get_current_user),
):
    """上传PDF，自动解析、切分、入库（含安全校验）"""
    # 安全校验1: 文件名
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "仅支持 PDF 文件")

    # 安全校验2: 文件大小上限50MB
    MAX_FILE_SIZE = 50 * 1024 * 1024
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(413, f"文件过大，上限 {MAX_FILE_SIZE // (1024*1024)}MB")

    # 安全校验3: PDF魔数 (magic bytes)
    if not content.startswith(b"%PDF"):
        raise HTTPException(400, "文件格式无效：不是合法的 PDF 文件")

    # 保存到data/ 目录（永久存储，重启后仍在）
    from src.config import DATA_DIR
    import hashlib
    os.makedirs(DATA_DIR, exist_ok=True)
    safe_name = file.filename.replace("\\", "_").replace("/", "_")
    pdf_path = DATA_DIR / safe_name
    content_hash = hashlib.md5(content).hexdigest()

    # 检查是否已存在相同文件（按内容哈希去重）
    for existing in DATA_DIR.glob("*.pdf"):
        if existing.stat().st_size == len(content):
            existing_hash = hashlib.md5(existing.read_bytes()).hexdigest()
            if existing_hash == content_hash:
                raise HTTPException(409, f"文件已存在: {existing.name}")  # 409 Conflict

    # 同名但不同内容的文件加序号
    counter = 1
    while pdf_path.exists():
        stem, ext = os.path.splitext(safe_name)
        pdf_path = DATA_DIR / f"{stem}_{counter}{ext}"
        counter += 1
    try:
        pdf_path.write_bytes(content)
    except Exception as e:
        raise HTTPException(500, f"文件保存失败: {e}")

    try:
        parser = DocumentParser()
        result = parser.load_pdf(str(pdf_path))
        if not result or not result.get("text"):
            raise HTTPException(422, "PDF 解析失败或内容为空")

        text = result["text"]
        page_count = result["page_count"]
        file_size = result["file_size_bytes"]

        doc_id = db.register_document(
            file_name=file.filename,
            file_path=str(pdf_path),
            file_size_bytes=file_size,
            page_count=page_count,
        )

        chunks = parser.split_text_semantically(text)
        stem = Path(file.filename).stem
        for i, chunk in enumerate(chunks):
            db.save_chunk(f"{stem}_chunk_{i:04d}", chunk, doc_id, chunk_index=i)

        db.update_document_status(doc_id, "completed")
        engine.refresh_index()

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"处理失败: {str(e)[:200]}")
    return {
        "doc_id": doc_id, "file_name": file.filename,
        "page_count": page_count, "file_size_bytes": file_size,
        "chunk_count": len(chunks), "status": "completed"
    }


@router.get("/documents", tags=["Documents"])
def list_documents(status: str = Query("completed"), db=Depends(get_db), user=Depends(get_current_user)):
    """列出知识库中的文档"""
    cursor = db.sqlite_conn.cursor()
    cursor.execute(
        "SELECT doc_id, file_name, page_count, file_size_bytes, parse_status, created_at "
        "FROM documents WHERE parse_status=? ORDER BY created_at DESC LIMIT 50",
        (status,))
    return [
        {"doc_id": r[0], "file_name": r[1], "page_count": r[2],
         "file_size_bytes": r[3], "parse_status": r[4], "created_at": r[5]}
        for r in cursor.fetchall()
    ]


@router.get("/documents/{doc_id}", tags=["Documents"])
def get_document(doc_id: str, db=Depends(get_db), user=Depends(get_current_user)):
    """获取文档详情"""
    doc = db.get_document(doc_id)
    if not doc:
        raise HTTPException(404, f"文档不存在: {doc_id}")

    cursor = db.sqlite_conn.cursor()
    cursor.execute(
        "SELECT chunk_id, chunk_index, char_count FROM chunks WHERE doc_id=? ORDER BY chunk_index",
        (doc_id,))
    doc["chunks"] = [{"chunk_id": r[0], "chunk_index": r[1], "char_count": r[2]}
                     for r in cursor.fetchall()]
    cursor.execute("SELECT COUNT(*) FROM qa_pairs WHERE doc_id=?", (doc_id,))
    doc["qa_count"] = cursor.fetchone()[0]
    return doc


@router.post("/search", tags=["Search"])
def search(req: SearchRequest, engine=Depends(get_engine), user=Depends(get_current_user)):
    """混合检索（语义+关键词+RRF+Reranker）"""
    results = engine.search(req.query, top_k=req.top_k)
    items = []
    for r in results:
        items.append({
            "chunk_id": r.chunk_id, "content": r.content,
            "source_file": r.source_file, "score": r.score,
            "vector_rank": r.vector_rank, "bm25_rank": r.bm25_rank,
            "qa_pairs": [{"q": qa["question"], "a": qa["answer"]} for qa in r.qa_pairs],
        })
    return {"query": req.query, "total": len(items), "results": items}


@router.post("/chat", tags=["Agent"])
def chat(req: ChatRequest, agent=Depends(get_agent), user=Depends(get_current_user)):
    """单轮Agent对话"""
    if req.reset:
        agent.reset()
    answer = agent.chat(req.message, verbose=False)
    return {"answer": answer}


@router.get("/stats", tags=["System"])
def get_stats(db=Depends(get_db), user=Depends(get_current_user)):
    """系统全局统计"""
    return db.get_stats()
