# QLoRA DPO 微调 Qwen3-8B
# 用法: python scripts/train_dpo.py
import sys, os, json, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import DPOTrainer
from datasets import Dataset

# 配置
MODEL_PATH = "E:/Document_Management_System/models/Qwen3-8B-Instruct"
DATA_DIR = Path(__file__).parent.parent / "train_data"
OUTPUT_DIR = Path(__file__).parent.parent / "models" / "Qwen3-8B-DPO"

def main():
    print("=" * 60)
    print("  QLoRA DPO 微调 Qwen3-8B")
    print("=" * 60)

    # 1. 加载数据
    print("[1/4] 加载训练数据...")
    with open(DATA_DIR / "dpo_train.json", "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(DATA_DIR / "dpo_test.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)
    print(f"  训练: {len(train_data)} 对 | 测试: {len(test_data)} 对")

    # 转换为 Dataset 格式
    train_dataset = Dataset.from_list([
        {"prompt": d["question"], "chosen": d["chosen"], "rejected": d["rejected"]}
        for d in train_data
    ])
    test_dataset = Dataset.from_list([
        {"prompt": d["question"], "chosen": d["chosen"], "rejected": d["rejected"]}
        for d in test_data
    ])

    # 2. 加载模型 (4-bit QLoRA)
    print("[2/4] 加载模型 (4-bit QLoRA)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    # A800: 可全精度训练，但不必要——4-bit QLoRA 已够

    print(f"  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem/1e9:.0f}GB")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # 3. LoRA 配置
    print("[3/4] 配置 LoRA...")
    peft_config = LoraConfig(
        r=32,       # A800: 更大LoRA秩
        alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 4. DPO 训练
    print("[4/4] DPO 训练...")
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=3,
        per_device_train_batch_size=4,     # A800: 大batch
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        save_total_limit=2,
        fp16=True,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        peft_config=peft_config,
        args=training_args,
        beta=0.1,
        max_length=2048,
        max_prompt_length=1024,
    )

    trainer.train()
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    print(f"\n[OK] 模型保存: {OUTPUT_DIR}")
    print("  下一步:")
    print(f"  python -m vllm.entrypoints.openai.api_server --model {OUTPUT_DIR} --port 8001")


if __name__ == "__main__":
    main()
