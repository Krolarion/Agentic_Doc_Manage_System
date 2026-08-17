# 系统评估编排器：测试集 → 检索 → Agent回答 → LLM Judge → 终端可视化报告
import json, time, os, random
from typing import List, Dict
from dataclasses import dataclass, field
from pathlib import Path

from src.storage.database_manager import DatabaseManager
from src.retrieval.search_engine import HybridSearchEngine
from src.agent.agent import DocumentAgent
from src.agent.tools import init_tools
from src.evaluation.judge import LLMJudge, JudgeScore

_BUILTIN_TESTS = [
    {"query": "合同审核需要注意哪些事项？"}, {"query": "机器学习和深度学习有什么关系？"},
    {"query": "敏捷开发的核心思想是什么？"}, {"query": "现金流量表的作用是什么？"},
    {"query": "如何提升员工的专业技能？"}, {"query": "会议纪要应该包含哪些内容？"},
]

C = {"R": "\033[91m", "G": "\033[92m", "Y": "\033[93m", "B": "\033[94m", "W": "\033[0m", "bold": "\033[1m"}


def load_from_dataset(n=None):
    """从test_data加载测试查询"""
    path = Path(__file__).parent.parent.parent / "test_data" / "qa_golden_testset.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        qa = json.load(f)["qa_pairs"]
    seen = set()
    unique = []
    for q in qa:
        if q["question"] not in seen:
            seen.add(q["question"])
            unique.append({"query": q["question"], "source_doc": q["source_doc"], "category": q.get("category","")})
    random.seed(42)
    if n:
        unique = random.sample(unique, min(n, len(unique)))
    return unique


def _bar(val, width=20, good=80):
    filled = int(val / 100 * width)
    bar_str = "|" * filled + " " * (width - filled)
    color = C["G"] if val >= good else C["Y"] if val >= 50 else C["R"]
    return f"{color}{bar_str}{C['W']}"


@dataclass
class EvalResult:
    query: str; answer: str; contexts: str; score: JudgeScore; latency_ms: float = 0.0
    source_doc: str = ""; category: str = ""
    def to_dict(self):
        return {"query": self.query, "answer": self.answer[:300],
                "scores": self.score.to_dict(), "latency_ms": self.latency_ms}


@dataclass
class EvalReport:
    results: List[EvalResult] = field(default_factory=list); model: str = ""
    def summary(self):
        if not self.results: return {}
        faith = [r.score.faithfulness for r in self.results]
        acc = [r.score.accuracy for r in self.results]
        rel = [r.score.relevance for r in self.results]
        ovr = [r.score.overall for r in self.results]
        n = len(self.results)
        by_cat = {}
        for r in self.results:
            cat = r.category or "unknown"
            if cat not in by_cat: by_cat[cat] = {"F":[],"A":[],"R":[],"O":[]}
            by_cat[cat]["F"].append(r.score.faithfulness)
            by_cat[cat]["A"].append(r.score.accuracy)
            by_cat[cat]["R"].append(r.score.relevance)
            by_cat[cat]["O"].append(r.score.overall)
        return {
            "model": self.model, "total": n,
            "F": f"{sum(faith)/n:.1f}%", "A": f"{sum(acc)/n:.1f}%",
            "R": f"{sum(rel)/n:.1f}%", "Overall": f"{sum(ovr)/n:.1f}%",
            "Pass80": f"{sum(1 for o in ovr if o>=80)/n*100:.0f}%",
            "by_category": {c: {k: f"{sum(v[k])/len(v[k]):.1f}%" for k in ["F","A","R","O"]} for c,v in by_cat.items()},
            "per_query": [r.to_dict() for r in self.results],
        }
    def to_json(self):
        return json.dumps(self.summary(), ensure_ascii=False, indent=2)


def _save_progress(report, total, done, elapsed_sec):
    """增量保存评估进度"""
    result_dir = Path(__file__).parent.parent.parent / "QAEval_result"
    result_dir.mkdir(exist_ok=True)

    faith = [r.score.faithfulness for r in report.results]
    acc = [r.score.accuracy for r in report.results]
    rel = [r.score.relevance for r in report.results]

    progress = {
        "status": "running",
        "progress": f"{done}/{total}",
        "current_F": f"{sum(faith)/len(faith):.1f}%",
        "current_A": f"{sum(acc)/len(acc):.1f}%",
        "current_R": f"{sum(rel)/len(rel):.1f}%",
        "elapsed_sec": int(elapsed_sec),
        "per_query": [r.to_dict() for r in report.results],
    }
    with open(result_dir / ".progress.json", "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


class SystemEvaluator:
    def __init__(self, db, engine):
        self.db = db; self.engine = engine; self.judge = LLMJudge()
        init_tools(db, engine); self.agent = DocumentAgent(backend="local")

    def run(self, test_cases=None, compact=None):
        cases = test_cases or _BUILTIN_TESTS
        report = EvalReport(model=self.judge.model)
        total = len(cases)
        if compact is None:
            compact = total > 10

        print(f"\n{C['bold']}{'='*70}{C['W']}")
        print(f"{C['bold']}  系统回答质量评估{C['W']}")
        print(f"  裁判: {self.judge.model} | 测试: {total} 条 | 模式: {'批量' if compact else '详细'}")
        print(f"{C['bold']}{'='*70}{C['W']}\n")

        if compact:
            print(f"  {'':>6s} {'查询':<32s} {'F':>4s} {'A':>4s} {'R':>4s} {'综合':>5s} {'延迟':>7s}")
            print(f"  {'':>6s} {'─'*32} {'─'*4} {'─'*4} {'─'*4} {'─'*5} {'─'*7}")

        t0 = time.time()
        for i, case in enumerate(cases):
            query = case.get("query", case.get("question", ""))
            src_doc = case.get("source_doc", "")
            cat = case.get("category", "")

            t1 = time.time()
            results = self.engine.search(query, top_k=5)
            ctx = "\n\n".join(r.content for r in results) if results else "(无)"
            self.agent.reset()
            answer = self.agent.chat(query, verbose=False)
            latency = (time.time() - t1) * 1000

            score = self.judge.evaluate(query, answer, ctx)
            report.results.append(EvalResult(query=query, answer=answer, contexts=ctx,
                                             score=score, latency_ms=latency, source_doc=src_doc, category=cat))

            if compact:
                qs = query[:30] + ".." if len(query) > 32 else query
                fc = f"{C['G']}{score.faithfulness:3d}%{C['W']}" if score.faithfulness >= 80 else f"{C['R']}{score.faithfulness:3d}%{C['W']}"
                print(f"  [{i+1:3d}/{total}] {qs:<32s} {fc} {score.accuracy:3d}% {score.relevance:3d}% {score.overall:4.1f}% {latency:6.0f}ms")
                if (i+1) % 50 == 0:
                    elapsed = time.time() - t0
                    avg_f = sum(r.score.faithfulness for r in report.results) / len(report.results)
                    eta = elapsed / (i+1) * (total - i - 1)
                    print(f"  {'─'*70}")
                    print(f"  [{i+1}/{total}] 当前F={avg_f:.0f}% | 已耗时{elapsed:.0f}s | 预计剩余{eta:.0f}s")
                    # 增量保存
                    _save_progress(report, total, i+1, elapsed)
                    print()
            else:
                print(f"  {C['bold']}[{i+1}/{total}]{C['W']} {query}")
                print(f"  {C['B']}回答:{C['W']} {answer[:120].strip()}{'...' if len(answer)>120 else ''}")
                print(f"  {C['Y']}理由:{C['W']} {score.reasoning[:100].strip()}...")
                print(f"  F:{score.faithfulness}% A:{score.accuracy}% R:{score.relevance}% O:{score.overall:.1f}% | {latency:.0f}ms\n")

        # 汇总
        s = report.summary()
        fv = float(s["F"].replace("%","")); av = float(s["A"].replace("%","")); rv = float(s["R"].replace("%",""))
        ov = float(s["Overall"].replace("%",""))
        print(f"\n  {C['bold']}{'='*70}{C['W']}")
        print(f"  {C['bold']}评估汇总 ({s['total']}条){C['W']}")
        print(f"  {'='*70}")
        print(f"  │ {'指标':<12s} │ {'得分':>8s} │ {'可视化':>30s} │")
        print(f"  ├{'─'*14}┼{'─'*10}┼{'─'*32}┤")
        for label, val in [("忠实度 F", (s["F"], fv)), ("准确率 A", (s["A"], av)), ("相关性 R", (s["R"], rv)), ("综合得分", (s["Overall"], ov))]:
            print(f"  │ {label:<12s} │ {val[0]:>8s} │ {_bar(val[1],30)} │")
        print(f"  ├{'─'*14}┴{'─'*10}┴{'─'*32}┤")
        print(f"  │ 80分以上: {s['Pass80']:<10s} │")
        print(f"  {'='*70}{C['W']}")

        # 按类别
        if s.get("by_category") and len(s["by_category"]) > 1:
            print(f"\n  {C['bold']}按文档类别:{C['W']}")
            print(f"  {'类别':<12s} {'F':>6s} {'A':>6s} {'R':>6s} {'O':>6s}")
            for cat, scores in sorted(s["by_category"].items()):
                print(f"  {cat:<12s} {scores['F']:>6s} {scores['A']:>6s} {scores['R']:>6s} {scores['O']:>6s}")

        # 版本化存储
        result_dir = Path(__file__).parent.parent.parent / "QAEval_result"
        result_dir.mkdir(exist_ok=True)
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version = f"v{len(list(result_dir.glob('eval_*.json'))) + 1:03d}"
        filename = f"eval_{version}_{timestamp}.json"
        path = result_dir / filename

        # 保存报告
        report_data = report.summary()
        report_data["version"] = version
        report_data["timestamp"] = timestamp
        report_data["model"] = self.judge.model
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)

        # 更新latest指针
        latest_path = result_dir / "latest.json"
        latest_path.write_text(filename, encoding="utf-8")

        # 保存版本索引
        index_path = result_dir / "index.json"
        index = []
        if index_path.exists():
            index = json.loads(index_path.read_text(encoding="utf-8"))
        index.append({"version": version, "file": filename, "timestamp": timestamp,
                       "queries": s["total"], "F": s["F"], "A": s["A"], "R": s["R"], "Overall": s["Overall"]})
        index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"\n  报告: QAEval_result/{filename}")
        print(f"  版本: {version}  |  历史: {len(index)} 次评估")
        return report
