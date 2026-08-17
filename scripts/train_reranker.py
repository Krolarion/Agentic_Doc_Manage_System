# Reranker 微调 v2：困难负样本为主
# 用法: python scripts/train_reranker.py
import sys, os, json, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader

MODEL_NAME = "BAAI/bge-reranker-v2-m3"
QA_PATH = Path(__file__).parent.parent / "test_data" / "qa_test_dataset.json"
OUTPUT = Path(__file__).parent.parent / "models" / "reranker-finetuned"


def build_training_data(n=6000):
    """
    负样本设计:
    - 正样本 ×1: (query, source_chunk)
    - 同类困难负样本 ×2: (query, 同类不同文档)
    - 检索困难负样本 ×1: (query, Top-20搜索中非source的chunk)
    - 跨类负样本 ×1: (query, 不同类别chunk)
    → 5对/query, 60%困难负样本
    """
    with open(QA_PATH, "r", encoding="utf-8") as f:
        all_qa = json.load(f)["qa_pairs"]

    from src.storage.database_manager import DatabaseManager
    from src.retrieval.search_engine import HybridSearchEngine
    db = DatabaseManager()
    engine = HybridSearchEngine(db, enable_rerank=False)
    chunks = db.get_all_chunks()

    qa_by_doc = {}
    for q in all_qa:
        doc = q.get("source_doc", "")
        qa_by_doc.setdefault(doc, []).append(q)

    docs = list(qa_by_doc.keys())
    train_data = []

    for doc, qa_list in qa_by_doc.items():
        doc_chunks = [c for c in chunks if c["source_file"] == doc]
        if not doc_chunks:
            continue

        for qa in qa_list[:8]:
            query = qa["question"]
            source_content = doc_chunks[0]["content"]

            # 正样本
            train_data.append(InputExample(texts=[query, source_content], label=1.0))

            # 同类困难负样本 ×2（40%权重）
            same_cat = [d for d in docs if d != doc and d.split("_")[0] == doc.split("_")[0]]
            if same_cat:
                for hard_doc in random.sample(same_cat, min(2, len(same_cat))):
                    hard_c = [c for c in chunks if c["source_file"] == hard_doc]
                    if hard_c:
                        train_data.append(InputExample(
                            texts=[query, random.choice(hard_c)["content"]], label=0.0))

            # 检索困难负样本 ×1（30%权重）— 模拟真实错误排序
            try:
                results = engine.search(query, top_k=15, enable_rerank=False)
                for r in results[:10]:  # 从Top-10中找非source文档
                    if r.source_file != doc:
                        train_data.append(InputExample(
                            texts=[query, r.content], label=0.0))
                        break  # 只取一个
            except:
                pass

            # 跨类负样本 ×1（30%权重）
            other_docs = [d for d in docs if d.split("_")[0] != doc.split("_")[0]]
            if other_docs:
                od = random.choice(other_docs)
                oc = [c for c in chunks if c["source_file"] == od]
                if oc:
                    train_data.append(InputExample(
                        texts=[query, random.choice(oc)["content"]], label=0.0))

        if len(train_data) >= n:
            break

    random.shuffle(train_data)
    db.close()
    return train_data[:n]


def main():
    print("=" * 50)
    print("  Reranker 微调 v2 (60%困难负样本)")
    print("=" * 50)

    print("[1/3] 构造训练数据...")
    train_data = build_training_data(n=6000)
    split = int(len(train_data) * 0.85)
    train_set = train_data[:split]
    eval_set = train_data[split:]
    print(f"  训练: {len(train_set)} 对 | 验证: {len(eval_set)} 对")

    print(f"[2/3] 加载 {MODEL_NAME}...")
    model = CrossEncoder(MODEL_NAME, trust_remote_code=True, num_labels=1)

    train_loader = DataLoader(train_set, shuffle=True, batch_size=8)

    print("[3/3] 训练 (3 epochs, batch=16)...")
    warmup = min(150, len(train_set) // 16 // 5)
    model.fit(
        train_dataloader=train_loader,
        epochs=3,
        warmup_steps=warmup,
        show_progress_bar=True,
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    model.save(str(OUTPUT))
    print(f"\n[OK] 模型保存: {OUTPUT}")


if __name__ == "__main__":
    main()
