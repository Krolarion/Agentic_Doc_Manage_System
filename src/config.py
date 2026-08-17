import os
from pathlib import Path

# 路径管理 (基于项目根目录动态解析)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SQLITE_DB_PATH = str(PROJECT_ROOT / "document_management.db")
CHROMA_DB_DIR = str(PROJECT_ROOT / "chroma_data")

# 加载 .env文件（本地开发用，生产环境应通过系统环境变量注入）
_ENV_FILE = PROJECT_ROOT / ".env"
if _ENV_FILE.exists():
    with open(_ENV_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


# LLM与API配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise RuntimeError(
        "DEEPSEEK_API_KEY 未设置。\n"
        "  方式1: 创建 .env 文件，写入 DEEPSEEK_API_KEY=你的key\n"
        "  方式2: 设置系统环境变量 set DEEPSEEK_API_KEY=你的key"
    )

DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "deepseek-v4-pro")

# 本地Qwen服务（Agent可选基座）
LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", "http://127.0.0.1:8001/v1")
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "Qwen2.5-7B-DPO-merged")


# 文档处理与切分配置
# 语义切分的最大字符阈值
CHUNK_SIZE = 600

# 嵌入(Embedding)模型配置
EMBEDDING_MODEL_NAME = str(PROJECT_ROOT / "models" / "embedding-finetuned")

# 重排序(Reranker)模型配置 — Cross-Encoder精排
RERANKER_MODEL_NAME = str(PROJECT_ROOT / "models" / "reranker-finetuned")


# 4. 抽取与裁判规则配置
# 裁判系统判断事实忠诚度时的最低温度 (保持0.0以保证结果稳定性)
CRITIC_TEMPERATURE = 0.0

# 知识抽取生成器使用的温度
GENERATOR_TEMPERATURE = 0.2
