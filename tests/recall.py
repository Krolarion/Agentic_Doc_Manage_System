import sys
import os
import json
import random

# 动态将当前文件所在目录的上一级（项目根目录）加入系统路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.storage.database_manager import DatabaseManager
from src.retrieval.search_engine import build_search_engine

def evaluate_json_recall(json_path: str, top_k: int = 5, sample_size: int = 300):
    print("========================================")
    print("启动纯检索引擎火力测试 (基于 Golden Testset)")
    print("========================================\n")
    
    if not os.path.exists(json_path):
        print(f"[错误] 找不到数据文件: {json_path}")
        return
        
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    qa_pairs = data.get("qa_pairs", [])
    total_available = len(qa_pairs)
    
    if total_available == 0:
        print("[错误] JSON 文件中没有数据。")
        return
        
    actual_sample_size = min(sample_size, total_available)
    eval_dataset = random.sample(qa_pairs, actual_sample_size)
    print(f"[OK] 随机抽取 {actual_sample_size} 条进行测试...\n")
    
    db = DatabaseManager()
    engine = build_search_engine(db)
    
    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    mrr_sum = 0.0
    valid_count = 0
    
    for i, item in enumerate(eval_dataset):
        # 使用洗好的 question 和 source_doc 进行严谨匹配
        query = item.get('question')
        true_source = item.get('source_doc')
        
        if not query or not true_source:
            true_source = item.get('doc_id')
            if not query or not true_source:
                continue
                
        valid_count += 1
        
        if valid_count % 50 == 0:
            print(f"  已评测 {valid_count}/{actual_sample_size} 条...")
            
        # 调用检索引擎
        results = engine.search(query, top_k=top_k)
        
        # 透视镜：打印前两笔的详细检索结果以便核查
        if valid_count <= 2:
            print(f"\n[透视镜 {valid_count}] Q: {query}")
            print(f"  👉 靶标文件 (Ground Truth): {true_source}")
            for rank, r in enumerate(results, 1):
                meta_str = str(getattr(r, 'metadata', ''))
                print(f"     Top {rank} 召回 -> doc_id: {r.doc_id}, chunk_id: {r.chunk_id} | 元数据: {meta_str[:60]}...")
            print("-" * 40)

        # 模糊匹配逻辑
        hit_rank = None
        true_source_clean = true_source.split('.')[0].lower()
        
        for rank, r in enumerate(results, start=1):
            retrieved_info = f"{r.doc_id} {r.chunk_id} {getattr(r, 'metadata', '')}".lower()
            if true_source_clean in retrieved_info:
                hit_rank = rank
                break
                
        # 计算得分
        if hit_rank is not None:
            mrr_sum += 1.0 / hit_rank
            if hit_rank == 1: hits_at_1 += 1
            if hit_rank <= 3: hits_at_3 += 1
            if hit_rank <= 5: hits_at_5 += 1

    if valid_count == 0:
        print("\n[错误] 仍然没有有效数据参与评估，请检查 JSON 字段。")
        return

    print("\n========== 检索引擎召回率成绩单 ==========")
    print(f"测试样本量 (Valid Queries): {valid_count}")
    print(f"Hit@1 (首条即命中)       : {hits_at_1 / valid_count * 100:.2f}%")
    print(f"Hit@3 (前三条包含答案)   : {hits_at_3 / valid_count * 100:.2f}%")
    print(f"Hit@5 (前五条包含答案)   : {hits_at_5 / valid_count * 100:.2f}%")
    print(f"MRR   (平均倒数排名)     : {mrr_sum / valid_count:.4f}")
    print("==========================================")

if __name__ == "__main__":
    # 指向刚刚生成的黄金测试集
    json_file_path = r"E:\Document_Management_System\Agentic_Doc_System\test_data\qa_golden_testset.json"
    evaluate_json_recall(json_path=json_file_path, top_k=5, sample_size=300)