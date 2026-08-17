# Qwen2.5-7B-Instruct OpenAI 兼容服务 (Windows) — 支持工具调用
# 用法: python scripts/run_qwen_server.py --port 8001
import sys, os, json, argparse, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

MODEL_PATH = "E:/Document_Management_System/Agentic_Doc_System/models/Qwen2.5-7B-SFT-merged"

app = FastAPI(title="Qwen2.5-7B-DPO API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

model = None
tokenizer = None


@app.get("/")
def root():
    return {"service": "Qwen2.5-7B-DPO-merged", "endpoints": ["/v1/models", "/v1/chat/completions"], "status": "ok"}


@app.get("/v1/models")
def list_models():
    return {"data": [{"id": "Qwen2.5-7B-DPO-merged", "object": "model"}]}


def _build_prompt(messages):
    text = ""
    for m in messages:
        role, content = m["role"], m["content"]
        if role == "system": text += f"<|im_start|>system\n{content}<|im_end|>\n"
        elif role == "user": text += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif role == "assistant": text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
    return text + "<|im_start|>assistant\n"


@app.post("/v1/chat/batch")
async def chat_batch(data: dict):
    """批量推理：一次 forward pass 处理多个 prompt"""
    batch = data.get("batch", [])  # [{"messages":[...], "temperature":0.8}, ...]
    if not batch:
        return {"responses": []}

    prompts = [_build_prompt(item["messages"]) for item in batch]
    temps = [item.get("temperature", 0.7) for item in batch]
    max_tok = batch[0].get("max_tokens", 600) if batch else 600

    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_tok,
            temperature=temps[0] if len(set(temps)) == 1 else 0.7,
            do_sample=(temps[0] > 0 if len(set(temps)) == 1 else True),
            pad_token_id=tokenizer.eos_token_id,
        )
    responses = []
    for i in range(len(prompts)):
        resp = tokenizer.decode(outputs[i][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        responses.append(resp)
    return {"responses": responses}


@app.post("/v1/chat/completions")
async def chat(data: dict):
    messages = data.get("messages", [])
    temperature = data.get("temperature", 0.7)
    max_tokens = data.get("max_tokens", 2048)
    tools = data.get("tools")

    # 构建 prompt（含工具定义）
    text = ""
    for m in messages:
        role, content = m.get("role"), m.get("content", "")
        if role == "system":
            text += f"<|im_start|>system\n{content}"
            # 注入工具定义到 system prompt
            if tools:
                tool_desc = json.dumps(tools, ensure_ascii=False)
                text += f"\n\n你可以使用以下工具:\n{tool_desc}"
                text += "\n要调用工具时，输出格式: <tool_call>{\"name\":\"工具名\",\"arguments\":{...}}</tool_call>"
            text += "<|im_end|>\n"
        elif role == "user":
            text += f"<|im_start|>user\n{content or ''}<|im_end|>\n"
        elif role == "assistant":
            # 显示工具调用内容
            tc = m.get("tool_calls")
            if tc:
                tc_text = json.dumps(tc, ensure_ascii=False)
                text += f"<|im_start|>assistant\n<tool_call>{tc_text}</tool_call>\n{content or ''}<|im_end|>\n"
            else:
                text += f"<|im_start|>assistant\n{content or ''}<|im_end|>\n"
        elif role == "tool":
            text += f"<|im_start|>tool\n{content or ''}<|im_end|>\n"
    text += "<|im_start|>assistant\n"

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    # 解析工具调用
    content = response
    tool_calls = None
    if "<tool_call>" in response:
        match = re.search(r'<tool_call>(.*?)</tool_call>', response, re.DOTALL)
        if match:
            try:
                tc = json.loads(match.group(1))
                tool_calls = [{
                    "id": "call_" + str(hash(tc.get("name","")) % 10000),
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": json.dumps(tc.get("arguments",{}))}
                }]
                content = response[:match.start()].strip()
            except:
                pass

    return {
        "choices": [{"message": {"role": "assistant", "content": content or "", "tool_calls": tool_calls}}],
        "model": "Qwen2.5-7B-Instruct",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--quantize", choices=["4bit", "8bit", "none"], default="4bit")
    args = parser.parse_args()

    global model, tokenizer

    model_name = Path(MODEL_PATH).name
    print(f"Loading {model_name} ({args.quantize})...")
    bnb = None
    if args.quantize == "4bit":
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif args.quantize == "8bit":
        bnb = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.padding_side = "left"  # decoder-only 必须左填充

    print(f"{model_name} ready at http://127.0.0.1:{args.port}")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
