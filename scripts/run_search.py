# 检索测试脚本
# 用法: python -m scripts.run_search "查询内容" 或 python -m scripts.run_search 进入交互模式
import sys

from src.storage.database_manager import DatabaseManager
from src.retrieval.search_engine import HybridSearchEngine


def search_loop(engine: HybridSearchEngine):
    """交互式检索"""
    print("\n交互检索模式 (输入 'quit' 退出, 'refresh' 重建索引)\n")
    while True:
        try:
            query = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            break

        if not query:
            continue
        if query.lower() == "quit":
            break
        if query.lower() == "refresh":
            engine.refresh_index()
            continue

        results = engine.search(query, top_k=5)

        if not results:
            print("  (无结果)\n")
            continue

        for i, r in enumerate(results):
            vec_info = f"vec#{r.vector_rank}" if r.vector_rank else "vec:-"
            bm25_info = f"bm25#{r.bm25_rank}" if r.bm25_rank else "bm25:-"
            print(f"  [{i + 1}] {r.source_file} (score={r.score:.4f}, {vec_info}, {bm25_info})")
            print(f"      {r.content[:100]}...")
            if r.qa_pairs:
                print(f"      [QA] {len(r.qa_pairs)}条 | {r.qa_pairs[0]['question'][:50]}")
            print()


def main():
    print("=" * 60)
    print("混合检索测试 (Vector + BM25 + RRF)")
    print("=" * 60)

    # 初始化
    db = DatabaseManager()
    engine = HybridSearchEngine(db)

    if len(sys.argv) > 1:
        # 单次查询模式
        query = " ".join(sys.argv[1:])
        print(f"\n查询: {query}\n")
        results = engine.search(query, top_k=5)
        if not results:
            print("(无结果)")
        for i, r in enumerate(results):
            print(f"[{i + 1}] {r.source_file} (score={r.score:.4f})")
            print(f"    {r.content[:120]}")
            if r.qa_pairs:
                print(f"    QA: {r.qa_pairs[0]['question']} → {r.qa_pairs[0]['answer'][:60]}")
            print()
    else:
        # 交互模式
        search_loop(engine)

    db.close()


if __name__ == "__main__":
    main()
