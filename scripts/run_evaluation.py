# 系统评估脚本
# 用法: python -m scripts.run_evaluation [--output report.json]
import os
import sys
import json

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from src.storage.database_manager import DatabaseManager
from src.retrieval.search_engine import HybridSearchEngine
from src.evaluation.evaluator import SystemEvaluator


def main():
    output_file = None
    if len(sys.argv) > 1:
        for i, arg in enumerate(sys.argv[1:]):
            if arg == "--output" and i + 2 < len(sys.argv):
                output_file = sys.argv[i + 2]
            elif arg == "--help":
                print("用法: python -m scripts.run_evaluation [--output report.json]")
                print("  --output <path>  将评估报告保存为 JSON 文件")
                return

    print("初始化系统...")
    db = DatabaseManager()
    engine = HybridSearchEngine(db, enable_rerank=True)

    stats = db.get_stats()
    print(f"知识库: {stats['documents']}文档 {stats['chunks']}Chunk {stats['qa_pairs']}QA\n")

    evaluator = SystemEvaluator(db, engine)
    report = evaluator.run(verbose=True)

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report.to_json())
        print(f"\n报告已保存: {output_file}")

    db.close()


if __name__ == "__main__":
    main()
