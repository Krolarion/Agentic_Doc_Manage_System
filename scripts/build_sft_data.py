# 构造 SFT 训练数据 (ReAct 对话格式)
# python scripts/build_sft_data.py
import sys, os, json, random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage.database_manager import DatabaseManager
from src.retrieval.search_engine import HybridSearchEngine

QA_PATH = Path(__file__).parent.parent / "test_data" / "qa_test_dataset.json"
OUTPUT = Path(__file__).parent.parent / "train_data" / "sft_data.json"

SYSTEM = "你是企业文档助手，回答前必须用 search_knowledge 检索相关信息。如找不到就老实说。"


def main():
    with open(QA_PATH, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)["qa_pairs"]

    db = DatabaseManager()
    engine = HybridSearchEngine(db, enable_rerank=False)

    # 筛选：优先取长问题（提高复杂度）
    qa_pairs.sort(key=lambda q: len(q["question"]), reverse=True)  # 长问题优先
    selected = qa_pairs[:6000]

    data = []
    for qa in selected:
        query = qa["question"]
        answer = qa["answer"]

        # 静默改写：去口语词提升检索质量
        stop_words = {"请","帮我","查询","搜索","找一下","有没有","是什么","什么意思","怎么回事","如何","怎样","什么是"}
        cleaned = query
        for w in stop_words:
            cleaned = cleaned.replace(w, " ")
        cleaned = " ".join(cleaned.split()) or query

        results = engine.search(cleaned, top_k=3)
        ctx = "\n".join(f"{r.source_file}: {r.content[:200]}" for r in results)

        # 合成复杂问题（前1000条中随机两两合并）
        composite_query = ""
        if len(query) < 30 and random.random() < 0.15:  # 15% 概率合并短问题
            other = random.choice(selected)
            composite_query = f"{query}。另外，{other['question']}"
            composite_answer = f"关于{query}：{answer}\n\n关于{other['question']}：{other['answer']}"
            # 重新检索合并后的 query
            comp_results = engine.search(composite_query, top_k=5)
            ctx = "\n".join(f"{r.source_file}: {r.content[:200]}" for r in comp_results)
            query = composite_query
            answer = composite_answer

        # 去掉 tool 角色（llama-factory 不支持），合并为 assistant 双轮
        data.append({"conversations": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": query},
            {"role": "assistant", "content": f"工具返回: {ctx[:1500]}"},   # 搜索结果
            {"role": "user", "content": "请基于以上搜索结果回答原问题"},
            {"role": "assistant", "content": answer},
        ]})

        if len(data) % 200 == 0:
            print(f"  [{len(data)}/6000]")

    # 质量过滤
    def valid(d):
        convs = d["conversations"]
        return (len(convs) == 5
                and all(c["content"] for c in convs)
                and len(convs[-1]["content"]) > 5)
    clean = [d for d in data if valid(d)]
    print(f"  过滤: {len(data)} → {len(clean)} ({len(data)-len(clean)} 条不合格)")

    # 数据分析
    q_lens = [len(d["conversations"][1]["content"]) for d in clean]
    a_lens = [len(d["conversations"][4]["content"]) for d in clean]
    q_simple = sum(1 for q in q_lens if q <= 15)           # 短问题(简单)
    q_medium = sum(1 for q in q_lens if 15 < q <= 30)     # 中等问题
    q_complex = sum(1 for q in q_lens if q > 30)           # 复杂问题
    # 语义多样性
    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer("BAAI/bge-small-zh-v1.5")
    prompts = [d["conversations"][1]["content"] for d in clean[:500]]
    embeddings = emb.encode(prompts)
    from sklearn.metrics.pairwise import cosine_similarity
    sims = cosine_similarity(embeddings)
    mask = ~np.eye(len(sims), dtype=bool)
    avg_sim = sims[mask].mean()

    print(f"\n  质量指标:")
    print(f"    Prompt 长度: {min(q_lens)}-{max(q_lens)}字 (均值{sum(q_lens)//len(q_lens)})")
    print(f"    答案长度:   {min(a_lens)}-{max(a_lens)}字 (均值{sum(a_lens)//len(a_lens)})")
    print(f"    难度分布:   简单{q_simple}条 / 中等{q_medium}条 / 复杂{q_complex}条")
    print(f"    语义相似度: {avg_sim:.3f} (<0.60高多样, >0.85需去重)")

    random.shuffle(clean)
    OUTPUT.parent.mkdir(exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] {len(clean)} 条 → {OUTPUT}")

    db.close()


if __name__ == "__main__":
    main()
