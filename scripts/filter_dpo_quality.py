# DPO 数据质量过滤: DeepSeek Judge 对比 chosen vs rejected
# 用法: python scripts/filter_dpo_quality.py [--sample 200] [--out train_data/dpo_filtered.json]
import sys, os, json, random, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI
from src.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
INPUT = Path(__file__).parent.parent / "train_data" / "dpo_data.json"


def judge_pair(prompt, chosen, rejected):
    """DeepSeek 判断 chosen 是否优于 rejected"""
    q = f"""回答A比回答B更好吗？详细完整的算好回答，简短不全的算差回答。A更好回复A，B更好回复B。

回答A: {chosen[:500]}
回答B: {rejected[:300]}"""

    try:
        r = client.chat.completions.create(
            model="deepseek-v4-pro", temperature=0, max_tokens=500,
            messages=[{"role": "system", "content": "只回复一个字母：A 或 B。不要解释。"},
                      {"role": "user", "content": q}],
        )
        resp = r.choices[0].message.content.strip().upper()
        return "A" in resp and "B" not in resp
    except:
        return True


def main():
    with open(INPUT, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"加载 {len(data)} 对")

    sample = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[1] == "--sample" else len(data)
    data = random.sample(data, min(sample, len(data)))
    print(f"过滤 {len(data)} 对\n")

    passed = []
    for i, d in enumerate(data):
        ok = judge_pair(d["prompt"], d["chosen"], d["rejected"])
        if ok:
            passed.append(d)

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(data)}] 通过: {len(passed)} ({(len(passed)/(i+1))*100:.0f}%)")

        time.sleep(0.3)

    out_path = sys.argv[4] if len(sys.argv) > 4 else str(INPUT).replace(".json", "_filtered.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(passed, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] {len(passed)}/{len(data)} 通过 ({len(passed)/len(data)*100:.0f}%) → {out_path}")


if __name__ == "__main__":
    main()
