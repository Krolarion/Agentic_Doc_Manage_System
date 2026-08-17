#!/usr/bin/env python
"""
企业文档智能管理系统 — 统一入口

用法:
  python Run.py server             启动 Web 服务 (日常使用只需这个)
  python Run.py ingest --all        批量入库 data/ 中所有 PDF
  python Run.py ingest --file xx    入库单个 PDF
  python Run.py search              命令行交互检索
  python Run.py chat                命令行 AI 对话
  python Run.py eval                运行系统评估 → evaluation_report.json
  python Run.py status              查看知识库状态
  python Run.py setup               首次运行：检查环境 + 入库所有文档
"""

import sys, os, argparse
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


# ═══════════════════════════════════════
# 命令实现
# ═══════════════════════════════════════

def cmd_server(args):
    """启动 Web 服务（一直运行，Ctrl+C 停止）"""
    from src.api.main import app
    import uvicorn

    print_banner("企业文档智能管理系统")
    print(f"  前端界面    http://127.0.0.1:{args.port}")
    print(f"  API 文档    http://127.0.0.1:{args.port}/docs")
    print(f"  登录账号    admin / admin123")
    print("=" * 50 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="info")


def cmd_ingest(args):
    """文档入库 — 解析 + 切分 + 向量化，可选 QA 生成"""
    from src.storage.database_manager import DatabaseManager
    from src.ingestion.document_parser import DocumentParser
    from src.retrieval.search_engine import HybridSearchEngine

    files = _resolve_files(args)
    if not files:
        print("[!] 没有找到 PDF 文件。放入 data/ 目录或使用 --file 指定。")
        return

    import src.config as cfg
    if args.chunk_size:
        cfg.CHUNK_SIZE = args.chunk_size

    db = DatabaseManager()
    engine = HybridSearchEngine(db, enable_rerank=False)
    parser = DocumentParser()
    total_chunks = 0

    for i, fp in enumerate(files):
        print(f"\n[{i+1}/{len(files)}] {fp.name}")
        result = parser.load_pdf(str(fp))
        if not result or not result.get("text"):
            print("  [SKIP] 解析失败")
            continue

        text = result["text"]
        pages = result["page_count"]
        fsize = result["file_size_bytes"]

        doc_id = db.register_document(fp.name, str(fp.resolve()), fsize, pages)
        result = parser.split_parent_child(text)
        parents = result["parents"]
        children = result["children"]
        # 存父块 + 子块
        for pid, pcontent in parents:
            db.save_chunk(f"{fp.stem}_{pid}", pcontent, doc_id, 0)
        for cid, pid, ccontent in children:
            db.save_chunk(f"{fp.stem}_{cid}", ccontent, doc_id, 0)
        db.update_document_status(doc_id, "completed")
        print(f"  [OK] {len(parents)}父块/{len(children)}子块 | {pages}页 | {fsize:,}字节")
        total_chunks += len(children)

    engine.refresh_index()

    qa_passed = []
    # 可选：自动生成 QA 对
    if args.with_qa and total_chunks > 0:
        print(f"\n[QA] 开始为 {total_chunks} 个 chunk 生成 QA 对...")
        from src.knowledge.qa_generator import QAGenerator
        from src.knowledge.qa_critic import FaithfulnessCritic, DiversityFilter, RelevancyFilter
        from src.agent.tools import init_tools
        from sentence_transformers import SentenceTransformer
        from src.config import EMBEDDING_MODEL_NAME

        init_tools(db, engine)
        generator = QAGenerator()
        embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

        cursor = db.sqlite_conn.cursor()
        cursor.execute("SELECT chunk_id, content, doc_id FROM chunks ORDER BY doc_id, chunk_index")
        all_chunks = cursor.fetchall()
        all_qa = []

        for ci in range(0, len(all_chunks), args.qa_batch):
            batch = all_chunks[ci:ci + args.qa_batch]
            batch_texts = [b[1] for b in batch]
            qa_list = generator.generate_from_chunks(batch_texts)
            for qa in qa_list:
                bi = qa.get("chunk_index", 0)
                if bi < len(batch):
                    qa["chunk_id"] = batch[bi][0]
                    qa["doc_id"] = batch[bi][2]
                    all_qa.append(qa)
            print(f"  [{ci+1}-{min(ci+args.qa_batch, len(all_chunks))}] 生成 {len(qa_list)} 条")

        # 三层过滤
        faith = FaithfulnessCritic()
        qa_passed = faith.filter(all_qa, [b[1] for b in all_chunks])
        div = DiversityFilter(embedder=embedder)
        qa_passed = div.filter(qa_passed)
        rel = RelevancyFilter(embedder=embedder)
        qa_passed = rel.filter(qa_passed, [b[1] for b in all_chunks])

        for qa in qa_passed:
            db.save_qa(
                qa_id=f"qa_{qa.get('chunk_id','')}_{hash(qa['question'])%10000:04d}",
                chunk_id=qa.get("chunk_id", ""),
                doc_id=qa.get("doc_id", ""),
                question=qa["question"],
                answer=qa["answer"],
                faith_score=qa.get("faith_score", 0),
                faith_reasoning=qa.get("faith_reasoning", ""),
                diversity_max_sim=qa.get("diversity_max_sim", 0),
                relevancy_score=qa.get("relevancy_score", 0),
                filter_status="passed",
            )
        print(f"  [QA] 入库 {len(qa_passed)}/{len(all_qa)} 条 (留存率 {len(qa_passed)/max(len(all_qa),1)*100:.0f}%)")

    db.close()
    stats = f"{total_chunks} chunks"
    if args.with_qa:
        stats += f", {len(qa_passed)} QA 对"
    print(f"\n[DONE] 入库 {len(files)} 篇文档 ({stats})")


def cmd_search(args):
    """命令行交互检索"""
    from src.storage.database_manager import DatabaseManager
    from src.retrieval.search_engine import HybridSearchEngine

    db = DatabaseManager()
    engine = HybridSearchEngine(db, enable_rerank=True)
    s = db.get_stats()
    print(f"知识库: {s['documents']}文档 {s['chunks']}chunk {s['qa_pairs']}QA | quit 退出\n")

    while True:
        try:
            q = input("search> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if q in ("quit", "exit", "q"): break
        if not q: continue
        results = engine.search(q, top_k=5)
        for i, r in enumerate(results):
            print(f"  [{i+1}] {r.source_file}  score={r.score:.4f}")
            print(f"       {r.content[:100]}")
        print()
    db.close()


def cmd_chat(args):
    """命令行 AI 对话"""
    from src.storage.database_manager import DatabaseManager
    from src.retrieval.search_engine import HybridSearchEngine
    from src.agent.agent import DocumentAgent
    from src.agent.tools import init_tools

    db = DatabaseManager()
    engine = HybridSearchEngine(db, enable_rerank=True)
    init_tools(db, engine)
    agent = DocumentAgent()
    s = db.get_stats()
    print(f"知识库: {s['documents']}文档 {s['chunks']}chunk | quit/stats\n")

    while True:
        try:
            q = input("chat> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if q in ("quit", "exit", "q"): break
        if not q: continue
        if q == "stats":
            s = db.get_stats()
            print(f"  文档{s['documents']} Chunk{s['chunks']} QA{s['qa_pairs']}")
            continue
        answer = agent.chat(q, verbose=False)
        print(f"\n{answer}\n")
    db.close()


def cmd_eval(args):
    """系统评估 — 用 LLM Judge 对 Agent 回答做 F/A/R 三维评分"""
    from src.storage.database_manager import DatabaseManager
    from src.retrieval.search_engine import HybridSearchEngine
    from src.evaluation.evaluator import SystemEvaluator, load_from_dataset

    db = DatabaseManager()
    engine = HybridSearchEngine(db, enable_rerank=True)
    evaluator = SystemEvaluator(db, engine)

    if args.dataset:
        test_cases = load_from_dataset(n=args.dataset)
        if not test_cases:
            # 🚀 修改这里：把 qa_test_dataset.json 改成 qa_golden_testset.json
            print("[!] test_data/qa_golden_testset.json 不存在，请检查数据清洗脚本是否运行成功")
            db.close(); return
        print(f"从数据集加载了 {len(test_cases)} 条测试查询\n")
    elif args.quick:
        test_cases = [{"query": "违约金怎么计算"}, {"query": "员工试用期多久"}, {"query": "项目预算情况"}]
    else:
        test_cases = None  # 使用内置6条

    report = evaluator.run(test_cases=test_cases)  # type: ignore
    db.close()


def cmd_status(args):
    """查看知识库状态"""
    from src.storage.database_manager import DatabaseManager
    db = DatabaseManager()
    s = db.get_stats()
    print_banner("知识库状态")
    print(f"  文档:   {s['documents']} 个  {_fmt_dict(s.get('doc_by_status',{}))}")
    print(f"  Chunk:  {s['chunks']} 个")
    print(f"  QA 对:  {s['qa_pairs']} 条  {_fmt_dict(s.get('qa_by_filter',{}))}")
    print(f"  审计:   {s['audit_log']} 条")

    cursor = db.sqlite_conn.cursor()
    cursor.execute("SELECT file_name, page_count, parse_status FROM documents ORDER BY created_at DESC LIMIT 10")
    docs = cursor.fetchall()
    if docs:
        print(f"\n  最近文档:")
        for name, pages, status in docs:
            print(f"    {name}  ({pages}页) [{status}]")
    db.close()
    print()


def cmd_setup(args):
    """首次运行向导"""
    print_banner("系统初始化")

    # 依赖
    deps = {"fitz":"PyMuPDF", "sentence_transformers":"sentence-transformers",
            "chromadb":"chromadb", "openai":"openai", "jieba":"jieba",
            "fastapi":"fastapi", "uvicorn":"uvicorn"}
    missing = []
    for mod, pkg in deps.items():
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"  [!] 缺少依赖: {' '.join(missing)}")
        print(f"  -> pip install {' '.join(missing)} python-multipart\n")
    else:
        print(f"  [OK] 依赖完整\n")

    # .env
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        print(f"  [OK] .env 已存在")
    else:
        key = input("  DeepSeek API Key: ").strip()
        if key:
            env_file.write_text(f"DEEPSEEK_API_KEY={key}\n", encoding="utf-8")
            print(f"  [OK] .env 已创建\n")

    # 文档
    pdfs = list((PROJECT_ROOT / "data").glob("*.pdf"))
    print(f"  data/ 中有 {len(pdfs)} 个 PDF")
    if pdfs:
        ans = input("  是否批量入库全部文档？(y/n): ").strip().lower()
        if ans == "y":
            cmd_ingest(argparse.Namespace(all=True, file=None, chunk_size=0, with_qa=False, qa_batch=3))

    print(f"\n[OK] 初始化完成。运行: python Run.py server")


# ═══════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════

def print_banner(title):
    print("\n" + "=" * 50)
    print(f"  {title}")
    print("=" * 50)

def _fmt_dict(d: dict) -> str:
    return " | ".join(f"{k}:{v}" for k, v in d.items())

def _resolve_files(args):
    pdf_dir = PROJECT_ROOT / "data"
    if args.all:
        return sorted(pdf_dir.glob("*.pdf"))
    elif args.file:
        return [Path(args.file)]
    else:
        # 默认取第一个
        files = sorted(pdf_dir.glob("*.pdf"))
        if files:
            print(f"[提示] 未指定文件，默认处理: {files[0].name}")
            print(f"       使用 --all 批量处理全部 {len(files)} 个文件\n")
        return files[:1]


# ═══════════════════════════════════════
# CLI
# ═══════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="企业文档智能管理系统 — DocBrain",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="日常使用: python Run.py server   →  浏览器打开 http://127.0.0.1:8000"
    )
    sub = parser.add_subparsers(dest="command", help="可用命令")

    p = sub.add_parser("server", help="启动 Web 服务 (前端+API)")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_server)

    p = sub.add_parser("ingest", help="文档入库 (解析+切分+QA蒸馏)")
    p.add_argument("--file", help="指定单个 PDF 文件")
    p.add_argument("--all", action="store_true", help="data/ 中所有 PDF")
    p.add_argument("--chunk-size", type=int, default=0, help="chunk 大小 (默认600)")
    p.add_argument("--with-qa", action="store_true", help="同时生成 QA 对并做三层蒸馏")
    p.add_argument("--qa-batch", type=int, default=3, help="QA 批量大小 (默认3)")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("search", help="命令行交互检索")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("chat", help="命令行 AI 对话")
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("eval", help="系统评估 F/A/R")
    p.add_argument("--dataset", type=int, default=0, metavar="N", help="从test_data抽取N条(推荐300-600)")
    p.add_argument("--quick", action="store_true", help="快速评估 (3条)")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("status", help="查看知识库状态")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("setup", help="首次运行向导")
    p.set_defaults(func=cmd_setup)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
