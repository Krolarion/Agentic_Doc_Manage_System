# Reranker 性能评估：对比 ON vs OFF
# 用法: python scripts/eval_reranker.py
import sys, os, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
from src.storage.database_manager import DatabaseManager
from src.retrieval.search_engine import HybridSearchEngine


def load_test_queries(n=200):
    """从 QA 数据集采样测试查询"""
    path = Path(__file__).parent.parent / "test_data" / "qa_test_dataset.json"
    if not path.exists():
        print("[!] test_data/qa_test_dataset.json 不存在"); return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    qa = data["qa_pairs"]
    # 去重（相同问题只保留一条）
    seen, unique = set(), []
    for q in qa:
        if q["question"] not in seen:
            seen.add(q["question"]); unique.append(q)
    np.random.seed(42)
    sample = np.random.choice(unique, min(n, len(unique)), replace=False)
    return list(sample)


def hit_metrics(results, source_doc, metrics):
    """判断结果集中是否命中 source_doc"""
    metrics["total"] += 1
    for rank, r in enumerate(results):
        if r["source_file"] == source_doc:
            metrics["hits"] += 1
            metrics["rr_sum"] += 1.0 / (rank + 1)
            dcg = 1.0 / np.log2(rank + 2)
            metrics["ndcg_sum"] += dcg / (1.0 / np.log2(2))
            return
    # 未命中


def evaluate_reranker(db, engine, test_queries, top_k=10):
    """核心评估：向量 / BM25 / RRF / RRF+Reranker 四种模式"""
    all_metrics = {
        "Vector":    {"hits": 0, "total": 0, "rr_sum": 0.0, "ndcg_sum": 0.0},
        "BM25":      {"hits": 0, "total": 0, "rr_sum": 0.0, "ndcg_sum": 0.0},
        "RRF":       {"hits": 0, "total": 0, "rr_sum": 0.0, "ndcg_sum": 0.0},
        "RRF+Re":    {"hits": 0, "total": 0, "rr_sum": 0.0, "ndcg_sum": 0.0},
    }

    print(f"\n评估 {len(test_queries)} 条查询 (top_k={top_k})...")

    for qi, qa in enumerate(test_queries):
        query = qa["question"]
        source_doc = qa["source_doc"]

        # 粗召回: 向量检索
        vec_raw = db.search_chunks(query, top_k)
        vec_results = [{"source_file": h["metadata"].get("source_file", "")} for h in vec_raw]
        hit_metrics(vec_results, source_doc, all_metrics["Vector"])

        # 粗召回: BM25
        if engine.bm25 and engine.chunks_index:
            tokens = engine._tokenize(query)
            bm25_raw = engine.bm25.search(tokens, top_k)
            bm25_results = []
            for idx, score in bm25_raw:
                if idx < len(engine.chunks_index):
                    bm25_results.append({"source_file": engine.chunks_index[idx]["source_file"]})
            hit_metrics(bm25_results, source_doc, all_metrics["BM25"])

        # RRF (无 Reranker)
        results_rrf = engine.search(query, top_k=top_k, enable_rerank=False)
        rrf_src = [{"source_file": r.source_file} for r in results_rrf]
        hit_metrics(rrf_src, source_doc, all_metrics["RRF"])

        # RRF + Reranker
        results_rerank = engine.search(query, top_k=top_k, enable_rerank=True)
        rerank_src = [{"source_file": r.source_file} for r in results_rerank]
        hit_metrics(rerank_src, source_doc, all_metrics["RRF+Re"])

        if (qi + 1) % 30 == 0:
            print(f"  [{qi+1}/{len(test_queries)}]")

    # 汇总
    summaries = {}
    for label, m in all_metrics.items():
        n = m["total"]
        summaries[label] = {
            "Recall": f"{m['hits']/n*100:.1f}%",
            "MRR": f"{m['rr_sum']/n:.4f}",
            "查询数": n,
        }
    return summaries


def main():
    print("=" * 60)
    print("  Reranker 性能评估")
    print("=" * 60)

    test_queries = load_test_queries(n=400)
    if not test_queries:
        return
    print(f"测试查询: {len(test_queries)} 条 (来自 test_data/qa_dataset.json)")

    db = DatabaseManager()
    engine = HybridSearchEngine(db, enable_rerank=True)
    print(f"知识库: {db.get_stats()['chunks']} chunks\n")

    t0 = time.time()
    summaries = evaluate_reranker(db, engine, test_queries, top_k=10)
    elapsed = time.time() - t0

    # 输出报告
    print("\n" + "=" * 60)
    print("  检索评估报告")
    print("=" * 60)
    print(f"  {'模式':<12s} {'Recall':>8s} {'MRR':>8s}")
    print(f"  {'-'*12} {'-'*8} {'-'*8}")
    for label, s in summaries.items():
        print(f"  {label:<12s} {s['Recall']:>8s} {s['MRR']:>8s}")

    rrf_recall = float(summaries["RRF"]["Recall"].replace("%",""))
    rerank_recall = float(summaries["RRF+Re"]["Recall"].replace("%",""))
    vec_recall = float(summaries["Vector"]["Recall"].replace("%",""))
    bm25_recall = float(summaries["BM25"]["Recall"].replace("%",""))

    print(f"\n  Reranker 提升: Recall {rerank_recall-rrf_recall:+.1f}% | MRR 对比见上表")
    print(f"  向量召回: {summaries['Vector']['Recall']} | BM25召回: {summaries['BM25']['Recall']}")
    print(f"  评估耗时: {elapsed:.0f}s")

    db.close()

    # 版本化存储
    from datetime import datetime
    result_dir = Path(__file__).parent.parent / "reranker_eval"
    result_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    v = f"v{len(list(result_dir.glob('eval_*.json'))) + 1:03d}"
    filename = f"eval_{v}_{ts}.json"
    report = {"summaries": summaries, "delta_rerank": f"{rerank_recall-rrf_recall:+.1f}%",
              "test_queries": len(test_queries), "version": v, "timestamp": ts}

    with open(result_dir / filename, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 更新索引
    index_path = result_dir / "index.json"
    idx = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else []
    idx.append({"version": v, "file": filename, "timestamp": ts, "Vector": summaries["Vector"]["Recall"], "BM25": summaries["BM25"]["Recall"], "RRF": summaries["RRF"]["Recall"], "RRF+Re": summaries["RRF+Re"]["Recall"]})
    index_path.write_text(json.dumps(idx, ensure_ascii=False, indent=2), encoding="utf-8")
    # latest 指针
    (result_dir / "latest.json").write_text(filename, encoding="utf-8")

    print(f"\n报告: reranker_eval/{filename}")
    print(f"版本: {v}  |  历史: {len(idx)} 次评估")


if __name__ == "__main__":
    main()
