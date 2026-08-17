"""
DPO 偏好数据构造管线 v2

改进:
- chosen:  DeepSeek V4 Pro (强模型)
- rejected: Qwen3-8B (待微调模型)
- 评分: F(40%) + A(20%) + R(20%) + 完整度(20%) 四维加权
- 过度安全型: 10-15%，仅用于明显有答案的 query
"""
import sys, os, json, time, random, hashlib, re
from pathlib import Path
from collections import Counter
from typing import List, Dict, Tuple
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# 加载 .env + 强制离线
env_file = Path(__file__).parent.parent / ".env"
if env_file.exists():
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from src.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, LLM_MODEL_NAME, EMBEDDING_MODEL_NAME
from src.storage.database_manager import DatabaseManager
from src.retrieval.search_engine import HybridSearchEngine
from src.agent.tools import init_tools

OUTPUT_DIR = Path(__file__).parent.parent / "train_data"
TARGET_SIZE = 5000   # A800: 目标恢复5000
MIN_DELTA = 0.3
SIM_THRESHOLD = 0.85
OVERSAFETY_RATIO = 0.125  # 过度安全型占负样本 12.5%

# 评分权重
SCORE_W = {"F": 0.4, "A": 0.2, "R": 0.2, "C": 0.2}  # F忠实/A准确/R相关/C完整度

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


def compare_pair(query: str, answer_a: str, answer_b: str, contexts: str) -> dict:
    prompt = f"""评估以下两个回答的质量。

用户问题: {query}
参考上下文: {contexts[:3000]}

回答A: {answer_a[:1500]}
回答B: {answer_b[:1500]}

分别对A和B进行四维评分:
- F (忠实度): 事实是否能在上下文找到支撑？
- A (准确率): 事实是否正确？
- R (相关性): 是否回应问题？
- C (完整度): 是否完整？

返回JSON:
{{"A_F":80,"A_A":75,"A_R":90,"A_C":70,
  "B_F":50,"B_A":85,"B_R":60,"B_C":40,
  "winner":"A|B|tie", "reasoning":"对比理由"}}"""

    try:
        r = client.chat.completions.create(
            model=LLM_MODEL_NAME, temperature=0.0,
            messages=[{"role": "system", "content": "严格输出JSON"}, {"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        result = json.loads(r.choices[0].message.content)
        a_f, a_a, a_r, a_c = result.get("A_F",0), result.get("A_A",0), result.get("A_R",0), result.get("A_C",0)
        b_f, b_a, b_r, b_c = result.get("B_F",0), result.get("B_A",0), result.get("B_R",0), result.get("B_C",0)
        a_w = a_f*SCORE_W["F"]+a_a*SCORE_W["A"]+a_r*SCORE_W["R"]+a_c*SCORE_W["C"]
        b_w = b_f*SCORE_W["F"]+b_a*SCORE_W["A"]+b_r*SCORE_W["R"]+b_c*SCORE_W["C"]
        return {
            "A": {"F":a_f,"A":a_a,"R":a_r,"C":a_c,"weighted":round(a_w,1)},
            "B": {"F":b_f,"A":b_a,"R":b_r,"C":b_c,"weighted":round(b_w,1)},
            "winner": result.get("winner","tie"), "reasoning": result.get("reasoning",""),
        }
    except:
        return {"A":{"F":0,"A":0,"R":0,"C":0,"weighted":0},"B":{"F":0,"A":0,"R":0,"C":0,"weighted":0},"winner":"error","reasoning":""}


class DeepSeekAgent:
    """强模型: DeepSeek V4 Pro API"""
    def __init__(self):
        self.client = client
        self.model = LLM_MODEL_NAME
        self.prompt = "你是企业文档助手。严格基于检索到的文档内容回答，引用来源，如实说明信息不足。"

    def chat(self, query: str, contexts: str) -> str:
        try:
            r = self.client.chat.completions.create(
                model=self.model, temperature=0.3,
                messages=[
                    {"role": "system", "content": self.prompt},
                    {"role": "user", "content": f"参考文档:\n{contexts[:3000]}\n\n问题: {query}"}
                ],
            )
            return r.choices[0].message.content or ""
        except:
            return ""


class QwenAgent:
    """待微调模型: Qwen3"""
    def __init__(self):
        self.client = OpenAI(api_key="local", base_url="http://127.0.0.1:8001/v1")
        self.model = "Qwen3-8B-Instruct"

    def chat(self, query: str, contexts: str) -> str:
        try:
            r = self.client.chat.completions.create(
                model=self.model, temperature=0.8,
                messages=[
                    {"role": "system", "content": "你是企业文档助手，帮用户查找信息。"},
                    {"role": "user", "content": f"参考:\n{contexts[:3000]}\n\n问题: {query}"}
                ],
            )
            return r.choices[0].message.content or ""
        except:
            return ""

    def batch_chat(self, batch: list) -> list:
        """批量推理：一次 API 调用处理多条（Qwen 服务器的 /v1/chat/batch）"""
        items = []
        for query, ctx in batch:
            items.append({"messages": [
                {"role": "system", "content": "你是企业文档助手，帮用户查找信息。"},
                {"role": "user", "content": f"参考:\n{ctx[:3000]}\n\n问题: {query}"}
            ], "temperature": 0.8})
        try:
            import requests
            r = requests.post("http://127.0.0.1:8001/v1/chat/batch", json={"batch": items}, timeout=300)
            return r.json().get("responses", [""]*len(batch))
        except:
            return [""]*len(batch)


def check_has_answer(query: str, contexts: str) -> bool:
    """LLM 判断上下文是否明显包含答案"""
    prompt = f"""判断以下问题是否能在给定上下文中找到明确答案。

问题: {query}
上下文: {contexts[:2000]}

返回JSON: {{"has_answer": true或false, "confidence": 0-100}}"""

    try:
        r = client.chat.completions.create(
            model=LLM_MODEL_NAME, temperature=0.0,
            messages=[{"role": "system", "content": "严格JSON"}, {"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        result = json.loads(r.choices[0].message.content)
        return result.get("has_answer", False) and result.get("confidence", 0) >= 60
    except:
        return True  # 不确定时默认为有答案


def classify_difficulty(qa: dict) -> str:
    """分类难度：不用JSON模式以加速响应"""
    prompt = f"""判断以下问题的难度等级。70%的问题应为medium，15%为easy，15%为hard。

问题: {qa['question']}

严格标准:
- easy: 仅当答案是可直引的短语/数字/日期。如"试用期几个月→3个月"
- medium: 需提取多条信息、组织语言。如"违约金如何计算""合同审核注意什么"
- hard: 需跨段落推理、多步对比。如"对比两个方案优缺点""什么条件下适用"

只回复一个词: easy 或 medium 或 hard"""

    try:
        r = client.chat.completions.create(
            model=LLM_MODEL_NAME, temperature=0.0, max_tokens=10,
            messages=[{"role": "system", "content": "你是难度标注专家。只回复一个词。"}, {"role": "user", "content": prompt}],
        )
        text = r.choices[0].message.content.strip().lower()
        for d in ["medium", "hard", "easy"]:
            if d in text: return d
        return "medium"
    except:
        return "medium"


def dedup(prompts: List[str]) -> List[int]:
    """语义去重，返回保留的索引列表"""
    print(f"\n[去重] {len(prompts)} 条...")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    emb = embedder.encode(prompts)
    sim = cosine_similarity(emb)
    keep, removed = [], set()
    for i in range(len(prompts)):
        if i in removed: continue
        keep.append(i)
        for j in range(i + 1, len(prompts)):
            if sim[i][j] > SIM_THRESHOLD:
                removed.add(j)
    print(f"  保留: {len(keep)} (移除 {len(removed)})")
    return keep


def main():
    print("=" * 60)
    print(f"  DPO 数据构造 v2 (目标: {TARGET_SIZE}条)")
    print(f"  chosen=DeepSeek V4 | rejected=Qwen3-8B")
    print("=" * 60)

    # 加载 QA
    path = Path(__file__).parent.parent / "test_data" / "qa_test_dataset.json"
    with open(path, "r", encoding="utf-8") as f:
        all_qa = json.load(f)["qa_pairs"]
    seen = set(); unique = []
    for q in all_qa:
        if q["question"].strip() not in seen:
            seen.add(q["question"].strip()); unique.append(q)
    random.seed(2026); random.shuffle(unique)
    unique = unique[:8000]

    # 分类难度（支持断点续跑）
    print(f"\n[1/5] 分类 {len(unique)} 条 Prompt...")
    OUTPUT_DIR.mkdir(exist_ok=True)
    checkpoint_path = OUTPUT_DIR / ".classify_progress.json"

    # 加载已有进度
    by_diff = {"easy": [], "medium": [], "hard": []}
    start_idx = 0
    if checkpoint_path.exists():
        try:
            saved = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if isinstance(saved, dict) and "easy" in saved and isinstance(saved["easy"], list):
                by_diff = {k: [qa for qa in unique if qa["question"] in set(p.get("question","") for p in v)]
                           for k, v in saved.items() if k != "_next_idx"}
                start_idx = saved.get("_next_idx", 0)
                print(f"  [续跑] 从 {start_idx}/{len(unique)} 继续...")
            else:
                print(f"  [WARN] 存盘格式不兼容，重新分类")
        except Exception:
            print(f"  [WARN] 存盘损坏，重新分类")

    # 规则分类（不用LLM，数据集问题太雷同，LLM分不出差异）
    hard_signals = ["对比","分析","为什么","如何选择","区别","哪个更","适用条件","什么情况下","怎么判断","关系"]
    easy_signals = ["多少","几个","什么时间","何时","在哪","日期","谁","多大","多久"]

    for i in range(start_idx, len(unique)):
        qa = unique[i]
        q = qa["question"]
        if any(w in q for w in hard_signals):
            d = "hard"
        elif len(q) <= 10 and any(w in q for w in easy_signals):
            d = "easy"
        else:
            d = "medium"
        by_diff[d].append(qa)
        if (i + 1) % 50 == 0:
            # 每50条存盘
            save = {k: [{"question": q["question"]} for q in v] for k, v in by_diff.items()}
            save["_next_idx"] = i + 1
            checkpoint_path.write_text(json.dumps(save, ensure_ascii=False), encoding="utf-8")
            print(f"  [{i+1}/{len(unique)}] E:{len(by_diff['easy'])} M:{len(by_diff['medium'])} H:{len(by_diff['hard'])} (已存盘)")

    print(f"  分类完成: E:{len(by_diff['easy'])} M:{len(by_diff['medium'])} H:{len(by_diff['hard'])}")

    # 初始化引擎
    print("\n[2/5] 初始化引擎...")
    db = DatabaseManager()
    engine = HybridSearchEngine(db, enable_rerank=True)  # A800:全开
    init_tools(db, engine)

    deepseek = DeepSeekAgent()
    qwen = QwenAgent()
    print(f"  DeepSeek API + Qwen3-8B(本地) 就绪")

    # 搜索在线程外做，无需锁

    # 生成偏好对
    print(f"\n[3/5] 生成偏好对...")
    target = {"easy": int(TARGET_SIZE * 0.15), "medium": int(TARGET_SIZE * 0.70), "hard": int(TARGET_SIZE * 0.15)}
    oversafety_count = 0
    max_oversafety = int(TARGET_SIZE * OVERSAFETY_RATIO)
    raw_pairs = []

    # 分阶段处理：DeepSeek 并行 → Qwen 串行批处理 → 对比评分并行
    API_PARALLEL = 300 
    raw_pairs = []
    oversafety_count = 0

    for diff, qa_list in by_diff.items():
        need = target[diff]
        pool = random.sample(qa_list, min(int(need * 1.3), len(qa_list)))
        print(f"\n  [{diff}] {len(pool)} 条")

        # 阶段 A: 预先搜索 → 并行调 DeepSeek 生成 chosen
        print(f" 阶段A: 预搜索 + 并行生成chosen...")
        candidates = []  # (qa, query, ctx, chosen_ans)
        for batch_start in range(0, len(pool), API_PARALLEL):
            batch = pool[batch_start:batch_start + API_PARALLEL]
            # 预搜索所有 query（不在线程里搜）
            pre_ctx = []
            for qa in batch:
                results = engine.search(qa["question"], top_k=5)
                pre_ctx.append("\n\n".join(r.content for r in results) if results else "(无)")
            # 并行生成 chosen（只调 API，不碰 GPU）
            def get_chosen(args):
                qa, ctx = args
                return (qa, qa["question"], ctx, deepseek.chat(qa["question"], ctx))
            with ThreadPoolExecutor(max_workers=API_PARALLEL) as ex:
                candidates.extend(list(ex.map(get_chosen, zip(batch, pre_ctx))))
            print(f"      [{min(batch_start+API_PARALLEL, len(pool))}/{len(pool)}]")

        # 阶段 B: 批量调 Qwen 生成 rejected（一次forward pass处理20条）
        BATCH_QWEN = 64       # A800 80G: 一次64条
        print(f"    阶段B: 批量生成rejected (Qwen3-8B, batch={BATCH_QWEN})...")
        for batch_start in range(0, len(candidates), BATCH_QWEN):
            batch = candidates[batch_start:batch_start + BATCH_QWEN]
            queries_ctxs = [(c[1], c[2]) for c in batch]
            responses = qwen.batch_chat(queries_ctxs)
            for j, resp in enumerate(responses):
                idx = batch_start + j
                c = candidates[idx]
                candidates[idx] = (c[0], c[1], c[2], c[3], resp)
            batch_end = min(batch_start + BATCH_QWEN, len(candidates))
            print(f"      [{batch_end}/{len(candidates)}]")

        # 阶段 C: 并行调 DeepSeek 对比评分
        print(f"    阶段C: 并行对比评分 (DeepSeek, {API_PARALLEL}并发)...")
        for batch_start in range(0, len(candidates), API_PARALLEL):
            batch = candidates[batch_start:batch_start + API_PARALLEL]
            def score_one(item):
                qa, query, ctx, chosen_ans, rejected_ans = item
                cmp = compare_pair(query, chosen_ans, rejected_ans, ctx)
                delta = cmp["A"]["weighted"] - cmp["B"]["weighted"]
                if delta >= MIN_DELTA:
                    return {
                        "prompt": query, "chosen": chosen_ans, "rejected": rejected_ans,
                        "chosen_scores": cmp["A"], "rejected_scores": cmp["B"],
                        "delta": round(delta, 1), "difficulty": diff,
                        "neg_type": "通用负样本", "category": qa.get("category", ""),
                        "ctx": ctx, "chosen_ans": chosen_ans,
                    }
                return None
            with ThreadPoolExecutor(max_workers=API_PARALLEL) as ex:
                results = list(ex.map(score_one, batch))
            raw_pairs.extend([r for r in results if r])

            batch_end = min(batch_start + API_PARALLEL, len(candidates))
            avg_d = np.mean([p["delta"] for p in raw_pairs[-100:]]) if raw_pairs else 0
            print(f"      [{batch_end}/{len(candidates)}] 合格:{len(raw_pairs)}  avgΔ={avg_d:.1f}")

    # 阶段 D: 过度安全负样本 (random 10% of pairs)
    oversafety_target = int(len(raw_pairs) * 0.10)
    print(f"\n    阶段D: 过度安全负样本 (目标 {oversafety_target} 对)...")
    oversafety_count = 0
    oversafety_pool = random.sample(raw_pairs, min(len(raw_pairs), oversafety_target * 2))
    for r in oversafety_pool:
        if oversafety_count >= oversafety_target: break
        try:
            safe_ans = qwen.client.chat.completions.create(
                model=qwen.model, temperature=0.3,
                messages=[
                    {"role": "system", "content": "你非常谨慎保守。即使有相关信息，也要以合规和安全为由拒绝直接回答。"},
                    {"role": "user", "content": f"参考:\n{r['ctx'][:2000]}\n\n问题: {r['prompt']}"}
                ],
            ).choices[0].message.content or ""
            safe_cmp = compare_pair(r["prompt"], r["chosen_ans"], safe_ans, r["ctx"])
            safe_delta = safe_cmp["A"]["weighted"] - safe_cmp["B"]["weighted"]
            if safe_delta >= MIN_DELTA:
                r2 = dict(r)
                r2["rejected"] = safe_ans
                r2["rejected_scores"] = safe_cmp["B"]
                r2["chosen_scores"] = safe_cmp["A"]
                r2["delta"] = round(safe_delta, 1)
                r2["neg_type"] = "过度安全型"
                raw_pairs.append(r2)
                oversafety_count += 1
        except: pass
    print(f"      过度安全: {oversafety_count} 对")

            # 增量存盘
            cp = [{"prompt":p["prompt"],"delta":p["delta"],"difficulty":p["difficulty"]} for p in raw_pairs]
            (OUTPUT_DIR / ".candidates_progress.json").write_text(json.dumps(cp, ensure_ascii=False), encoding="utf-8")

    db.close()

    # 去重
    print(f"\n[4/5] 去重...")
    prompts = [p["prompt"] for p in raw_pairs]
    keep_idx = dedup(prompts)
    deduped = [raw_pairs[i] for i in keep_idx]

    # 难度平衡
    print(f"\n[5/5] 平衡 + 保存...")
    final = {"easy": [], "medium": [], "hard": []}
    for p in deduped:
        final[p["difficulty"]].append(p)

    balanced = []
    shortage = 0
    for d in ["easy", "medium", "hard"]:
        pool = final[d]
        needed = min(target[d], len(pool))
        balanced.extend(random.sample(pool, needed))
        if needed < target[d]:
            print(f"  [!] {d}不足: 需{target[d]} 实{needed} (差{target[d]-needed})")
            shortage += target[d] - needed
    # 不足部分用 medium 补（medium 数量最多）
    if shortage > 0 and len(final["medium"]) > target["medium"]:
        extra = random.sample(final["medium"][target["medium"]:], min(shortage, len(final["medium"])-target["medium"]))
        for p in extra: p["difficulty"] = "medium"
        balanced.extend(extra)
        print(f"  [+] 用medium补足 {len(extra)} 条")
    random.shuffle(balanced)

    # 指标
    deltas = [p["delta"] for p in balanced]
    neg_dist = Counter(p["neg_type"] for p in balanced)
    diff_dist = Counter(p["difficulty"] for p in balanced)
    all_prompts_str = " ".join(p["prompt"] for p in balanced)
    diversity = len(set(all_prompts_str)) / max(len(all_prompts_str), 1)

    metrics = {
        "total": len(balanced), "avg_delta": round(np.mean(deltas), 2),
        "blurry_rate": f"{sum(1 for d in deltas if d < 0.5)/len(deltas)*100:.1f}%",
        "diversity": round(diversity, 4),
        "difficulty": dict(diff_dist), "neg_types": dict(neg_dist),
        "chosen_model": "DeepSeek-V4-Pro", "rejected_model": "Qwen3-8B-Instruct",
    }

    # 保存
    OUTPUT_DIR.mkdir(exist_ok=True)
    split = int(len(balanced) * 0.9)
    for name, data in [("train", balanced[:split]), ("test", balanced[split:])]:
        formatted = []
        for p in data:
            formatted.append({
                "prompt": p["prompt"],
                "chosen": p["chosen"], "rejected": p["rejected"],
                "chosen_scores": p["chosen_scores"],
                "rejected_scores": p["rejected_scores"],
                "delta": p["delta"], "difficulty": p["difficulty"],
                "neg_type": p["neg_type"],
            })
        with open(OUTPUT_DIR / f"dpo_{name}.json", "w", encoding="utf-8") as f:
            json.dump(formatted, f, ensure_ascii=False, indent=2)

    with open(OUTPUT_DIR / "dpo_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"  [DONE]")
    print(f"  总对: {len(balanced)} | Δavg: {metrics['avg_delta']} | 多样性: {metrics['diversity']}")
    print(f"  负样本: {dict(neg_dist)}")
    print(f"  训练: train_data/dpo_train.json ({len(balanced[:split])}对)")
    print(f"  测试: train_data/dpo_test.json ({len(balanced[split:])}对)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
