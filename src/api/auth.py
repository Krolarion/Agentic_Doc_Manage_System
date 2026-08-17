# JWT认证模块
import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional
from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# 简易JWT（避免额外依赖PyJWT）
import json
import base64
from pathlib import Path

def _load_or_create_secret() -> str:
    """从文件加载持久化密钥，不存在则生成并保存。避免重启后token失效。"""
    key_file = Path(__file__).parent.parent.parent / ".secret_key"
    if key_file.exists():
        return key_file.read_text().strip()
    key = secrets.token_hex(32)
    key_file.write_text(key)
    return key

SECRET_KEY = _load_or_create_secret()
TOKEN_EXPIRE_HOURS = 24

security = HTTPBearer(auto_error=False)


def _b64_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64_decode(s: str) -> bytes:
    s += "=" * (4 - len(s) % 4)
    return base64.urlsafe_b64decode(s)


def _sign(payload: str) -> str:
    return hashlib.sha256(f"{payload}:{SECRET_KEY}".encode()).hexdigest()[:32]


def create_token(username: str, user_id: int, role: str) -> str:
    """生成JWT token"""
    now = datetime.now(timezone.utc)
    payload = json.dumps({
        "sub": username,
        "uid": user_id,
        "role": role,
        "iat": now.isoformat(),
        "exp": (now + timedelta(hours=TOKEN_EXPIRE_HOURS)).isoformat(),
    })
    header = _b64_encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    body = _b64_encode(payload.encode())
    signature = _sign(f"{header}.{body}")
    return f"{header}.{body}.{signature}"


def verify_token(token: str) -> Optional[dict]:
    """验证token，返回payload或None"""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        header, body, sig = parts
        if _sign(f"{header}.{body}") != sig:
            return None
        payload = json.loads(_b64_decode(body))
        exp = datetime.fromisoformat(payload["exp"])
        if datetime.now(timezone.utc) > exp:
            return None
        return payload
    except Exception:
        return None


def hash_password(password: str) -> str:
    return hashlib.sha256(f"doc_sys_salt:{password}".encode()).hexdigest()


def verify_password(password: str, password_hash: str) -> bool:
    return hash_password(password) == password_hash


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
):
    """依赖注入：从请求中验证并返回当前用户"""
    token = None
    if credentials:
        token = credentials.credentials
    else:
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            token = auth[7:]

    if not token:
        raise HTTPException(401, "未提供认证令牌")

    payload = verify_token(token)
    if not payload:
        raise HTTPException(401, "令牌无效或已过期")

    return payload


def require_admin(user=Depends(get_current_user)):
    """需要管理员角色"""
    if user.get("role") != "admin":
        raise HTTPException(403, "需要管理员权限")
    return user
