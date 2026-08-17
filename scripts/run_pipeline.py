# QA 生成与蒸馏 —— 编排脚本
import uuid
import re
from typing import List, Dict
from pathlib import Path

from src.config import EMBEDDING_MODEL_NAME
from src.ingestion.document_parser import DocumentParser
from src.storage.database_manager import DatabaseManager
from src.knowledge.qa_generator import QAGenerator
from src.knowledge.qa_critic import FaithfulnessCritic, DiversityFilter, RelevancyFilter
from sentence_transformers import SentenceTransformer

# 简易关键词 → 标签映射（后续可改为 LLM 自动分类）
_KEYWORD_TAGS = {
    "技术|编程|代码|算法|架构|系统": ("技术文档", "领域"),
    "合同|协议|条款|法律|合规": ("法律文书", "领域"),
    "财务|报表|预算|审计|发票": ("财务文档", "领域"),
    "人事|招聘|绩效|员工|考勤": ("人力资源", "领域"),
    "产品|需求|设计|方案|规划": ("产品文档", "类型"),
    "会议|纪要|决议|讨论": ("会议记录", "类型"),
    "报告|分析|调研|总结": ("研究报告", "类型"),
}


def _auto_tag(doc_id: str, text: str, db: DatabaseManager) -> List[int]:
    """根据文本内容自动匹配标签"""
    tag_ids = []
    for pattern, (tag_name, category) in _KEYWORD_TAGS.items():
        if re.search(pattern, text):
            tid = db.add_tag(tag_name, category)
            tag_ids.append(tid)
    if tag_ids:
        db.tag_document(doc_id, tag_ids)
    return tag_ids


def run(pdf_path: str, batch_size: int = 5):
    """
    全链路执行：
    PDF 解析 -> 注册文档 -> 切分 -> 入库 -> QA 生成 -> 三层蒸馏 -> 入库
    全程写入审计日志和过滤元数据。
    """
    run_id = f"run_{uuid.uuid4().hex[:12]}"
    print("=" * 60)
    print(f"[*] Run ID: {run_id}")
    print(f"[*] 开始处理: {pdf_path}")
    print("=" * 60)

    # ═══════════════════════════════════════════════
    # 阶段 1：文档解析 + 注册
    # ═══════════════════════════════════════════════
    print("\n[阶段 1/5] 文档解析与注册...")
    parser = DocumentParser()
    result = parser.load_pdf(pdf_path)

    if not result or not result.get("text"):
        print("[ERROR] 文档内容为空或解析失败，终止流程。")
        return

    text = result["text"]
    page_count = result["page_count"]
    file_size = result["file_size_bytes"]
    file_name = Path(pdf_path).name

    db = DatabaseManager()
    db.log_event(run_id, "parse_start", details={"file": file_name})

    doc_id = db.register_document(
        file_name=file_name,
        file_path=str(Path(pdf_path).resolve()),
        file_size_bytes=file_size,
        page_count=page_count,
    )
    print(f"  [OK] 文档已注册: {doc_id} ({page_count}页, {file_size:,}字节)")

    # 自动标签
    tag_ids = _auto_tag(doc_id, text, db)
    if tag_ids:
        print(f"  [OK] 自动标签: {tag_ids}")

    # ═══════════════════════════════════════════════
    # 阶段 2：语义切分
    # ═══════════════════════════════════════════════
    print("\n[阶段 2/5] 语义切分...")
    chunks = parser.split_text_semantically(text)
    print(f"  [OK] 切分完成，共 {len(chunks)} 个 Chunk")
    db.log_event(run_id, "chunk_done", doc_id=doc_id,
                 details={"chunk_count": len(chunks)})

    # ═══════════════════════════════════════════════
    # 阶段 3：Chunk 双写入库
    # ═══════════════════════════════════════════════
    print("\n[阶段 3/5] Chunk 入库 (SQLite + ChromaDB)...")
    source_base = Path(pdf_path).stem

    for i, chunk in enumerate(chunks):
        chunk_id = f"{source_base}_chunk_{i:04d}"
        db.save_chunk(chunk_id, chunk, doc_id, chunk_index=i)
    print(f"  [OK] {len(chunks)} 个 Chunk 已存入双库引擎")
    db.log_event(run_id, "chunk_save", doc_id=doc_id,
                 details={"count": len(chunks)})

    # ═══════════════════════════════════════════════
    # 阶段 4：批量 QA 生成
    # ═══════════════════════════════════════════════
    print("\n[阶段 4/5] LLM 批量生成 QA 对...")
    generator = QAGenerator()
    all_qa_pairs: List[Dict] = []

    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start:batch_start + batch_size]
        print(f"  -> 批次 [{batch_start}..{batch_start + len(batch) - 1}] ({len(batch)} chunks)...")

        qa_pairs = generator.generate_from_chunks(batch)
        for qa in qa_pairs:
            qa["chunk_index"] = qa.get("chunk_index", 0) + batch_start

        print(f"    本批生成 {len(qa_pairs)} 条 QA")
        all_qa_pairs.extend(qa_pairs)

    print(f"  [OK] 共生成 {len(all_qa_pairs)} 条原始 QA 对")
    db.log_event(run_id, "qa_generate", doc_id=doc_id,
                 details={"raw_count": len(all_qa_pairs)})

    if not all_qa_pairs:
        print("[WARN] 没有生成任何 QA 对，终止流程。")
        db.update_document_status(doc_id, "completed")
        db.log_event(run_id, "pipeline_end", doc_id=doc_id,
                     details={"result": "no_qa"}, status="warning")
        db.close()
        return

    # ═══════════════════════════════════════════════
    # 阶段 5：三层蒸馏 + 入库
    # ═══════════════════════════════════════════════
    print("\n[阶段 5/5] 三层质量蒸馏...")

    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # Layer 1: 忠实度
    print(f"\n  [Layer 1] 忠实度校验 (LLM Judge) -- 输入 {len(all_qa_pairs)} 条...")
    faith_critic = FaithfulnessCritic()
    qa_pairs = faith_critic.filter(all_qa_pairs, chunks)
    dropped_faith = len(all_qa_pairs) - len(qa_pairs)
    print(f"    通过: {len(qa_pairs)}, 丢弃: {dropped_faith}")
    db.log_event(run_id, "filter_faith", doc_id=doc_id,
                 details={"input": len(all_qa_pairs), "passed": len(qa_pairs),
                          "dropped": dropped_faith})

    # Layer 2: 多样性
    if qa_pairs:
        before_div = len(qa_pairs)
        print(f"\n  [Layer 2] 多样性去重 (Vector) -- 输入 {before_div} 条...")
        div_filter = DiversityFilter(embedder=embedder)
        qa_pairs = div_filter.filter(qa_pairs)
        dropped_div = before_div - len(qa_pairs)
        print(f"    通过: {len(qa_pairs)}, 丢弃: {dropped_div}")
        db.log_event(run_id, "filter_diversity", doc_id=doc_id,
                     details={"input": before_div, "passed": len(qa_pairs),
                              "dropped": dropped_div})

    # Layer 3: 相关性
    if qa_pairs:
        before_rel = len(qa_pairs)
        print(f"\n  [Layer 3] 答案相关性校验 (Vector) -- 输入 {before_rel} 条...")
        rel_filter = RelevancyFilter(embedder=embedder)
        qa_pairs = rel_filter.filter(qa_pairs, chunks)
        dropped_rel = before_rel - len(qa_pairs)
        print(f"    通过: {len(qa_pairs)}, 丢弃: {dropped_rel}")
        db.log_event(run_id, "filter_relevancy", doc_id=doc_id,
                     details={"input": before_rel, "passed": len(qa_pairs),
                              "dropped": dropped_rel})

    # 最终入库（含完整过滤元数据）
    print(f"\n  [SAVE] 最终入库 {len(qa_pairs)} 条 QA 对...")
    for qa in qa_pairs:
        chunk_idx = qa.get("chunk_index", -1)
        chunk_id = f"{source_base}_chunk_{chunk_idx:04d}" if chunk_idx >= 0 else "unknown"
        qa_id = f"qa_{uuid.uuid4().hex[:12]}"

        # 确定 filter_status
        filter_status = "passed"

        db.save_qa(
            qa_id=qa_id,
            chunk_id=chunk_id,
            doc_id=doc_id,
            question=qa["question"],
            answer=qa["answer"],
            faith_score=qa.get("faith_score", 0.0),
            faith_reasoning=qa.get("faith_reasoning", ""),
            diversity_max_sim=qa.get("diversity_max_sim", 0.0),
            relevancy_score=qa.get("relevancy_score", 0.0),
            filter_status=filter_status,
        )

    db.update_document_status(doc_id, "completed")
    db.log_event(run_id, "pipeline_end", doc_id=doc_id,
                 details={"final_qa_count": len(qa_pairs)})

    # ═══════════════════════════════════════════════
    # 汇总统计
    # ═══════════════════════════════════════════════
    stats = db.get_stats()
    db.close()

    print("\n" + "=" * 60)
    print("[DONE] 处理完毕! 汇总统计:")
    print(f"   Run ID:           {run_id}")
    print(f"   文档:              {file_name} ({page_count}页, {file_size:,}字节)")
    print(f"   Chunk 总数:        {len(chunks)}")
    print(f"   QA 生成总数:       {len(all_qa_pairs)}")
    print(f"   最终入库:           {len(qa_pairs)}")
    print(f"   总体留存率:         {len(qa_pairs) / max(len(all_qa_pairs), 1) * 100:.1f}%")
    print(f"   DB 统计:           文档{stats['documents']}, Chunk{stats['chunks']}, "
          f"QA{stats['qa_pairs']}, 日志{stats['audit_log']}")
    print("=" * 60)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python -m scripts.run_pipeline <pdf_path> [batch_size]")
        print("示例: python -m scripts.run_pipeline data/sample.pdf 5")
        sys.exit(1)

    pdf_path = sys.argv[1]
    batch = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    run(pdf_path, batch_size=batch)
