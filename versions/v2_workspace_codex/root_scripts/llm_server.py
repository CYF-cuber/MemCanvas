#!/usr/bin/env python3
"""
Lightweight OpenAI-compatible API server using transformers + FastAPI.
Serves Qwen2.5-VL-7B-Instruct for text-only requests (judge, Mem0, A-Mem).

Usage:
    CUDA_VISIBLE_DEVICES=1 python llm_server.py --port 8100
"""
import argparse, json, time, uuid
from threading import Lock

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

app = FastAPI()
model = None
tokenizer = None
gen_lock = Lock()

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = ""
    messages: list[Message]
    temperature: float = 0.7
    max_tokens: int = 1000
    response_format: Optional[dict] = None

class ChatChoice(BaseModel):
    index: int = 0
    message: Message
    finish_reason: str = "stop"

class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    prompt_messages = [{"role": m.role, "content": m.content} for m in req.messages]
    text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)

    with gen_lock:
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=min(req.max_tokens, 2000),
                do_sample=req.temperature > 0.01,
                temperature=max(req.temperature, 0.01) if req.temperature > 0.01 else 1.0,
            )
        response_text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

    return ChatResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
        created=int(time.time()),
        model=req.model or "qwen2.5-vl-7b",
        choices=[ChatChoice(message=Message(role="assistant", content=response_text))],
    )

@app.get("/v1/models")
async def list_models():
    return {"data": [{"id": "qwen2.5-vl-7b", "object": "model"}]}

@app.get("/health")
async def health():
    return {"status": "ok"}

def main():
    global model, tokenizer
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--model", default="/home/cyf/Qwen2.5-VL-7B-Instruct")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    from transformers import AutoTokenizer, AutoConfig
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    cfg = AutoConfig.from_pretrained(args.model)
    if "vl" in cfg.model_type.lower() or "vl" in args.model.lower():
        from transformers import Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="auto")
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="auto")
    print(f"Model loaded, type={cfg.model_type}")

    uvicorn.run(app, host="0.0.0.0", port=args.port)

if __name__ == "__main__":
    main()
