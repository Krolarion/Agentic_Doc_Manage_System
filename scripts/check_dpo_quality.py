# 检查 DPO 数据质量
# 用法: python scripts/check_dpo_quality.py
import json, random
from pathlib import Path

p = Path(__file__).parent.parent / "train_data" / "dpo_data.json"
d = json.loads(p.read_text(encoding="utf-8"))
c_l = [len(x["chosen"]) for x in d]
r_l = [len(x["rejected"]) for x in d]

print(f"总对: {len(d)}")
print(f"chosen:   {min(c_l)}-{max(c_l)} 字 (均值{sum(c_l)//len(c_l)})")
print(f"rejected: {min(r_l)}-{max(r_l)} 字 (均值{sum(r_l)//len(r_l)})")
print()

for i in random.sample(range(len(d)), 5):
    x = d[i]
    print(f"=== #{i} ===")
    print(f"CHOSEN({len(x['chosen'])}): {x['chosen'][:150]}")
    print(f"REJECTED({len(x['rejected'])}): {x['rejected'][:120]}")
    print()
