# Embedding 微调：用 QA 数据集做域适配
# 用法: python scripts/train_embedding.py
import sys, os, json, random
from pathlib import Path
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).parent.parent))

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

MODEL_NAME = "BAAI/bge-small-zh-v1.5"
QA_PATH = Path(__file__).parent.parent / "test_data" / "qa_test_dataset.json"
OUTPUT = Path(__file__).parent.parent / "models" / "embedding-finetuned"


def build_triplets(n=5000):
    with open(QA_PATH, "r", encoding="utf-8") as f:
        all_qa = json.load(f)["qa_pairs"]

    from src.storage.database_manager import DatabaseManager
    db = DatabaseManager()
    chunks = db.get_all_chunks()
    db.close()

    chunk_pool = [c for c in chunks if len(c["content"]) > 50]

    train_data = []
    for qa in all_qa:
        query = qa["question"]
        src_doc = qa.get("source_doc", "")
        # 找到源文档的 chunk
        src_chunks = [c for c in chunk_pool if c["source_file"] == src_doc]
        if not src_chunks:
            continue
        # 找不同文档的 chunk 做负样本
        neg_chunks = [c for c in chunk_pool if c["source_file"] != src_doc]
        if not neg_chunks:
            continue

        pos = random.choice(src_chunks)["content"]
        neg = random.choice(neg_chunks)["content"]

        train_data.append(InputExample(texts=[query, pos, neg]))

        if len(train_data) >= n:
            break

    random.shuffle(train_data)
    return train_data


def main():
    print("=" * 50)
    print("  Embedding 微调")
    print("=" * 50)

    print("[1/3] 构造三元组...")
    data = build_triplets(n=5000)
    split = int(len(data) * 0.85)
    print(f"  训练: {split} 三元组 | 验证: {len(data)-split}")

    print(f"[2/3] 加载 {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    train_loader = DataLoader(data[:split], shuffle=True, batch_size=16)

    print("[3/3] 训练 (MultipleNegativesRankingLoss)...")
    loss = losses.MultipleNegativesRankingLoss(model)
    model.fit(
        train_objectives=[(train_loader, loss)],
        epochs=2,
        warmup_steps=100,
        output_path=str(OUTPUT),
        show_progress_bar=True,
    )

    print(f"\n[OK] 模型: {OUTPUT}")


if __name__ == "__main__":
    main()
