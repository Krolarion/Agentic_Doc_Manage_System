# API 服务启动脚本
# 用法: python -m scripts.run_server [--port 8000] [--reload]
#      或 python scripts/run_server.py [--port 8000] [--reload]
import sys, os
from pathlib import Path

# 确保项目根目录在 sys.path 中（兼容直接 python xxx.py 的运行方式）
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse


def main():
    parser = argparse.ArgumentParser(description="企业文档智能管理系统 API 服务")
    parser.add_argument("--host", default="127.0.0.1", help="绑定地址 (默认 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="端口 (默认 8000)")
    parser.add_argument("--reload", action="store_true", help="开发模式热重载")
    args = parser.parse_args()

    print("=" * 50)
    print(f"  企业文档智能管理系统 API")
    url = f"http://127.0.0.1:{args.port}" if args.host in ("0.0.0.0", "127.0.0.1") else f"http://{args.host}:{args.port}"
    print(f"  地址: {url}")
    print(f"  文档: {url}/docs")
    print("=" * 50)

    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
