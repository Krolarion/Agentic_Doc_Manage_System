from peft import PeftModel
from transformers import AutoModelForCausalLM
import torch

print("合并 DPO LoRA 权重...")
base = AutoModelForCausalLM.from_pretrained(
    "E:/Document_Management_System/Agentic_Doc_System/models/Qwen2.5-7B-SFT-merged",
    torch_dtype=torch.bfloat16, device_map="auto"
)
model = PeftModel.from_pretrained(base, "E:/Document_Management_System/Agentic_Doc_System/models/Qwen2.5-7B-DPO")
model = model.merge_and_unload()
model.save_pretrained("E:/Document_Management_System/Agentic_Doc_System/models/Qwen2.5-7B-DPO-merged")
print("done")
