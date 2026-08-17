# DPO 数据构造 v3: Qwen批量 + 无API (纯本地, ~45分钟)
# 用法: python scripts/build_dpo_sft.py
import sys, os, json, random, time, requests
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

QA_PATH = Path(__file__).parent.parent / "test_data" / "qa_test_dataset.json"
OUTPUT = Path(__file__).parent.parent / "train_data" / "dpo_data.json"
QWEN_BATCH = "http://127.0.0.1:8001/v1/chat/batch"
BATCH_SIZE = 32

SYSTEM_CHOSEN = "你基于搜索结果详细回答。必须引用来源文件名。"
SYSTEM_REJECTED = "简单回答，不引用来源。"


def text_similarity(a, b):
    shorter = min(len(a), len(b))
    if shorter == 0: return 1.0
    return sum(1 for i in range(shorter) if a[i] == b[i]) / shorter


def batch_generate(items, system_prompt, temperature):
    """批量调 Qwen 生成"""
    batch = [{"messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"搜索结果:\n{ctx}\n\n问题: {query}"} if ctx else {"role": "user", "content": query},
    ], "temperature": temperature, "max_tokens": 800} for query, ctx in items]
    try:
        r = requests.post(QWEN_BATCH, json={"batch": batch}, timeout=180)
        return r.json().get("responses", [""] * len(batch))
    except Exception as e:
        print(f"  [WARN] batch失败: {e}")
        return [""] * len(batch)


def main():
    with open(QA_PATH, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)["qa_pairs"]

    # 增量：加载已有数据，跳过已处理的 query
    existing_queries = set()
    existing_data = []
    if OUTPUT.exists():
        existing_data = json.loads(OUTPUT.read_text(encoding="utf-8"))
        for d in existing_data:
            # 从 prompt 中提取原问题
            if "问题:" in d.get("prompt", ""):
                existing_queries.add(d["prompt"].split("问题: ")[-1].strip())
    print(f"已有 {len(existing_data)} 对, 跳过 {len(existing_queries)} 个已处理 query")

    from src.storage.database_manager import DatabaseManager
    from src.retrieval.search_engine import HybridSearchEngine
    db = DatabaseManager()
    engine = HybridSearchEngine(db, enable_rerank=False)

    # 预搜索 (跳过已处理的)
    items = []
    for qa in qa_pairs:
        if qa["question"] in existing_queries:
            continue
        if len(items) >= 1000:  # 每批1000条
            break
        query = qa["question"]
        results = engine.search(query, top_k=3)
        ctx = "\n".join(f"{r.source_file}: {r.content[:200]}" for r in results)
        if len(ctx) >= 50:
            items.append((query, ctx))
    db.close()

    print(f"候选: {len(items)} 条, 批大小: {BATCH_SIZE}")
    t0 = time.time()

    # 阶段1: 批量生成 chosen (低温度)
    chosen_results = []
    for i in range(0, len(items), BATCH_SIZE):
        batch = items[i:i + BATCH_SIZE]
        responses = batch_generate(batch, SYSTEM_CHOSEN, 0.2)
        chosen_results.extend(responses)
        print(f"  chosen: [{min(i + BATCH_SIZE, len(items))}/{len(items)}]")

    # 阶段2: 批量生成 rejected (高温度, 有ctx但不用)
    rejected_items = [(q, ctx) for q, ctx in items]
    rejected_results = []
    for i in range(0, len(rejected_items), BATCH_SIZE):
        batch = rejected_items[i:i + BATCH_SIZE]
        responses = batch_generate(batch, SYSTEM_REJECTED, 0.9)
        rejected_results.extend(responses)
        print(f"  rejected: [{min(i + BATCH_SIZE, len(items))}/{len(items)}]")

    # 阶段3: 过滤
    data, filtered_sim, filtered_short = [], 0, 0
    for idx in range(len(items)):
        query, ctx = items[idx]
        chosen = chosen_results[idx] if idx < len(chosen_results) else ""
        rejected = rejected_results[idx] if idx < len(rejected_results) else ""

        if len(chosen) < 15 or len(rejected) < 5:
            filtered_short += 1; continue
        if len(chosen) < len(rejected):
            filtered_short += 1; continue  # chosen必须比rejected长
        if text_similarity(chosen, rejected) > 0.70:
            filtered_sim += 1; continue

        # prompt 包含完整ReAct流程: 用户问题 + 工具调用 + 搜索结果
        full_prompt = f"用户问题: {query}\n<tool_call>search_knowledge({query[:80]})</tool_call>\n搜索结果:\n{ctx}\n\n请回答:"
        data.append({"prompt": full_prompt, "chosen": chosen, "rejected": rejected})

    elapsed = (time.time() - t0) / 60
    print(f"\n{'='*50}")
    print(f"  DPO 质量报告")
    print(f"  {'='*50}")
    print(f"  候选: {len(items)}")
    print(f"  合格: {len(data)} ({(len(data)/max(len(items),1))*100:.0f}%)")
    print(f"  相似过滤: {filtered_sim}")
    print(f"  过短过滤: {filtered_short}")
    print(f"  耗时: {elapsed:.0f} 分钟")

    # 追加到已有数据
    all_data = existing_data + data
    OUTPUT.parent.mkdir(exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] 新增 {len(data)} 对, 总计 {len(all_data)} 对 → {OUTPUT}")


if __name__ == "__main__":
    main()
