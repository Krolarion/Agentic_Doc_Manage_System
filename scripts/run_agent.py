# 智能文档助手 —— 交互式对话
# 用法: python -m scripts.run_agent
import sys
import io

# 强制 UTF-8 输出避免 Windows GBK 乱码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from src.storage.database_manager import DatabaseManager
from src.retrieval.search_engine import HybridSearchEngine
from src.agent.agent import DocumentAgent
from src.agent.tools import init_tools


def main():
    print("=" * 60)
    print("  企业文档智能助手 (Agent)")
    print("  基于 ReAct + 混合检索 + DeepSeek V4 Pro")
    print("=" * 60)

    # 初始化底层引擎
    print("\n[1/3] 连接数据库...")
    db = DatabaseManager()

    print("\n[2/3] 构建检索引擎...")
    engine = HybridSearchEngine(db, enable_rerank=True)

    # 注入到工具层
    init_tools(db, engine)

    print("\n[3/3] 启动 Agent...")
    agent = DocumentAgent()

    stats = db.get_stats()
    print(f"\n知识库就绪: {stats['documents']} 个文档, {stats['chunks']} 个 Chunk, {stats['qa_pairs']} 条 QA")
    print("\n输入 'quit' 退出, 'reset' 重置对话, 'stats' 查看统计\n")

    while True:
        try:
            user_input = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            break
        if user_input.lower() == "reset":
            agent.reset()
            print("[OK] 对话已重置")
            continue
        if user_input.lower() == "stats":
            s = db.get_stats()
            print(f"文档{s['documents']} | Chunk{s['chunks']} | QA{s['qa_pairs']} | 日志{s['audit_log']}")
            continue

        # Agent 思考 + 回答
        answer = agent.chat(user_input, verbose=True)
        print(f"\n{'=' * 60}")
        print(answer)
        print("=" * 60)

    db.close()


if __name__ == "__main__":
    main()
