# CMA-ES Prompt 优化脚本
# 用法: python -m scripts.run_rl_optimizer
import os, sys, io
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from src.storage.database_manager import DatabaseManager
from src.retrieval.search_engine import HybridSearchEngine
from src.rl.prompt_optimizer import PromptOptimizer

# 测试集：覆盖知识库内容和边界场景
TEST_QUERIES = [
    "合同审核需要注意什么？",
    "机器学习在人工智能中处于什么地位？",
    "敏捷开发的核心思想是什么？",
    "现金流量表的作用是什么？",
    "如何提升员工的专业技能？",
    "会议纪要应该记录哪些内容？",
    "公园里有什么花？",                  # 知识库外的内容
    "介绍一下量子计算的基本原理",          # 完全不在知识库中
]


def main():
    print("初始化系统...")
    db = DatabaseManager()
    engine = HybridSearchEngine(db, enable_rerank=True)

    stats = db.get_stats()
    print(f"知识库: {stats['documents']}文档 {stats['chunks']}Chunk\n")

    # 优化
    optimizer = PromptOptimizer(
        db=db, engine=engine,
        pop_size=12,          # 种群大小（小批量快速验证）
        generations=8,        # 进化代数
        elite_ratio=0.3,
    )

    best = optimizer.run(test_queries=TEST_QUERIES, verbose=True)

    # 对比
    print("\n" + "=" * 60)
    print("优化前后对比:")
    comparison = optimizer.compare(TEST_QUERIES)
    orig = comparison["original"]
    opt = comparison["optimized"]
    print(f"  原始:   F={orig['F']}  A={orig['A']}  R={orig['R']}")
    print(f"  优化后: F={opt['F']}  A={opt['A']}  R={opt['R']}")

    # 保存最优 prompt
    prompt_path = "optimized_prompt.txt"
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(comparison["best_prompt"])
    print(f"\n最优 prompt 已保存: {prompt_path}")

    best_pct = float(opt['F'].replace('%',''))
    orig_pct = float(orig['F'].replace('%',''))
    improvement = best_pct - orig_pct
    print(f"忠实度提升: {improvement:+.1f}%")

    db.close()


if __name__ == "__main__":
    main()
