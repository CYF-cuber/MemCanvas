#!/usr/bin/env python3
"""
LoCoMo-10 QA experiment with/without memory retrieval.
- Baseline: question only
- Memory: retrieve top-k dialogue turns (CLIP text) and add as context
Outputs results and a report under /home/cyf/codex.
"""
import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

import torch
from transformers import CLIPProcessor, CLIPModel
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from sentence_transformers import SentenceTransformer

DATA_PATH_DEFAULT = "/home/cyf/codex/datasets/locomo/locomo10.json"
MODEL_QWEN_DEFAULT = "/home/cyf/Qwen2.5-VL-7B-Instruct"
CLIP_MODEL = "openai/clip-vit-large-patch14"
SEMANTIC_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_ROOT = "/home/cyf/codex"


@dataclass
class MemoryTurn:
    sample_id: str
    dia_id: str
    speaker: str
    text: str


class CLIPTextEncoder:
    def __init__(self, model_name: str = CLIP_MODEL):
        # Force offline to avoid network calls.
        self.processor = CLIPProcessor.from_pretrained(model_name, local_files_only=True)
        self.model = CLIPModel.from_pretrained(model_name, local_files_only=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.model.eval()

    def encode_texts(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        all_embeds = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encode texts"):
            batch = texts[i:i+batch_size]
            inputs = self.processor(text=batch, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                feats = self.model.get_text_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_embeds.append(feats.cpu().numpy())
        return np.concatenate(all_embeds, axis=0) if all_embeds else np.zeros((0, 768), dtype=np.float32)


class QwenAnswerer:
    def __init__(self, model_path: str = MODEL_QWEN_DEFAULT):
        self.model_path = model_path
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()

    def answer(self, question: str, context: List[MemoryTurn]) -> str:
        prompt_parts = []
        if context:
            prompt_parts.append("You are given memory snippets from a conversation. Use them to answer the question.")
            for i, m in enumerate(context):
                prompt_parts.append(f"Memory {i+1} [{m.speaker}]: {m.text}")
            prompt_parts.append("---")
        prompt_parts.append("Answer the question briefly and directly.")
        prompt_parts.append(f"Question: {question}")
        prompt = "\n".join(prompt_parts)

        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=32, do_sample=False)
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.processor.decode(generated_ids, skip_special_tokens=True).strip()
        return response


def normalize(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_match(pred: str, gold: str) -> bool:
    p = normalize(pred)
    g = normalize(gold)
    if not p or not g:
        return False
    if p == g:
        return True
    if g in p or p in g:
        return True
    return False


def semantic_match_scores(preds: List[str], golds: List[str], threshold: float) -> Tuple[float, int]:
    if not preds:
        return 0.0, 0
    # Force offline to avoid network calls.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    model = SentenceTransformer(
        SEMANTIC_MODEL,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    pred_emb = model.encode(preds, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
    gold_emb = model.encode(golds, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
    sims = (pred_emb * gold_emb).sum(axis=1)
    correct = int((sims >= threshold).sum())
    acc = (correct / len(preds)) * 100.0
    return acc, correct


def parse_conversation(item: Dict) -> List[MemoryTurn]:
    conv = item.get("conversation", {})
    sample_id = item.get("sample_id", "")
    session_keys = [k for k in conv.keys() if k.startswith("session_") and not k.endswith("date_time")]
    def session_idx(k: str) -> int:
        try:
            return int(k.split("_")[1])
        except Exception:
            return 0
    session_keys.sort(key=session_idx)

    turns: List[MemoryTurn] = []
    for sk in session_keys:
        for msg in conv.get(sk, []):
            turns.append(MemoryTurn(
                sample_id=sample_id,
                dia_id=msg.get("dia_id", ""),
                speaker=msg.get("speaker", ""),
                text=msg.get("text", ""),
            ))
    return turns


def build_memory(data: List[Dict], encoder: CLIPTextEncoder) -> Dict[str, Dict]:
    mem = {}
    for item in data:
        sample_id = item.get("sample_id", "")
        turns = parse_conversation(item)
        texts = [f"{t.speaker}: {t.text}" for t in turns]
        embeds = encoder.encode_texts(texts) if texts else np.zeros((0, 768), dtype=np.float32)
        mem[sample_id] = {
            "turns": turns,
            "embeddings": embeds,
        }
    return mem


def retrieve_top_k(question: str, mem_entry: Dict, encoder: CLIPTextEncoder, top_k: int) -> List[Tuple[int, float]]:
    q_emb = encoder.encode_texts([question])
    if mem_entry["embeddings"].shape[0] == 0:
        return []
    sims = mem_entry["embeddings"] @ q_emb.T
    sims = sims.squeeze(1)
    top_idx = np.argsort(-sims)[:top_k]
    return [(int(i), float(sims[i])) for i in top_idx]


def compute_memory_size(mem: Dict[str, Dict]) -> Dict:
    text_bytes = 0
    embed_bytes = 0
    turns_count = 0
    for sample_id, entry in mem.items():
        turns = entry["turns"]
        turns_count += len(turns)
        for t in turns:
            txt = f"{t.speaker}: {t.text}"
            text_bytes += len(txt.encode("utf-8"))
            # minimal metadata
            text_bytes += len(t.dia_id.encode("utf-8")) + len(sample_id.encode("utf-8"))
        embed_bytes += entry["embeddings"].nbytes
    return {
        "turns": turns_count,
        "text_bytes": text_bytes,
        "embedding_bytes": embed_bytes,
        "total_bytes": text_bytes + embed_bytes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DATA_PATH_DEFAULT)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--output-root", default=OUTPUT_ROOT)
    parser.add_argument("--qwen-model", default=MODEL_QWEN_DEFAULT)
    parser.add_argument("--max-qa", type=int, default=0, help="0 means all")
    parser.add_argument("--semantic-threshold", type=float, default=0.7)
    args = parser.parse_args()

    out_dir = Path(args.output_root) / f"locomo_experiment_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.data, "r", encoding="utf-8") as f:
        data = json.load(f)

    encoder = CLIPTextEncoder()
    mem = build_memory(data, encoder)

    size_stats = compute_memory_size(mem)

    answerer = QwenAnswerer(args.qwen_model)

    baseline_correct = 0
    memory_correct = 0
    total = 0
    evidence_hit = 0
    base_preds: List[str] = []
    mem_preds: List[str] = []
    golds: List[str] = []

    results = []

    for item in data:
        sample_id = item.get("sample_id", "")
        qas = item.get("qa", [])
        if args.max_qa and total >= args.max_qa:
            break
        for qa in qas:
            if args.max_qa and total >= args.max_qa:
                break
            question = qa.get("question", "")
            answer = qa.get("answer", "")
            evidence = set(qa.get("evidence", []))

            mem_entry = mem[sample_id]
            top = retrieve_top_k(question, mem_entry, encoder, args.top_k)
            retrieved_turns = [mem_entry["turns"][idx] for idx, _ in top]
            if evidence:
                if any(t.dia_id in evidence for t in retrieved_turns):
                    evidence_hit += 1

            pred_base = answerer.answer(question, [])
            pred_mem = answerer.answer(question, retrieved_turns)

            base_ok = is_match(pred_base, answer)
            mem_ok = is_match(pred_mem, answer)

            baseline_correct += int(base_ok)
            memory_correct += int(mem_ok)
            total += 1
            base_preds.append(pred_base)
            mem_preds.append(pred_mem)
            golds.append(str(answer))

            results.append({
                "sample_id": sample_id,
                "question": question,
                "answer": answer,
                "pred_baseline": pred_base,
                "pred_memory": pred_mem,
                "baseline_correct": base_ok,
                "memory_correct": mem_ok,
                "evidence": list(evidence),
                "retrieved_dia_ids": [t.dia_id for t in retrieved_turns],
            })

    baseline_acc = (baseline_correct / total * 100.0) if total else 0.0
    memory_acc = (memory_correct / total * 100.0) if total else 0.0
    evidence_hit_rate = (evidence_hit / total * 100.0) if total else 0.0
    semantic_base_acc, semantic_base_correct = semantic_match_scores(
        base_preds, golds, args.semantic_threshold
    )
    semantic_mem_acc, semantic_mem_correct = semantic_match_scores(
        mem_preds, golds, args.semantic_threshold
    )

    summary = {
        "total": total,
        "top_k": args.top_k,
        "baseline": {"correct": baseline_correct, "accuracy": baseline_acc},
        "memory": {"correct": memory_correct, "accuracy": memory_acc},
        "semantic": {
            "threshold": args.semantic_threshold,
            "baseline_correct": semantic_base_correct,
            "baseline_accuracy": semantic_base_acc,
            "memory_correct": semantic_mem_correct,
            "memory_accuracy": semantic_mem_acc,
        },
        "evidence_hit_rate": evidence_hit_rate,
        "memory_size": size_stats,
    }

    (out_dir / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "predictions.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    # report
    report = []
    report.append("# LoCoMo-10 实验报告")
    report.append("")
    report.append(f"- QA 数量: {total}")
    report.append(f"- Top-K: {args.top_k}")
    report.append("")
    report.append("## 准确率")
    report.append(f"- Baseline (无记忆): {baseline_acc:.2f}% ({baseline_correct}/{total})")
    report.append(f"- Memory (检索记忆): {memory_acc:.2f}% ({memory_correct}/{total})")
    report.append("")
    report.append("## 语义匹配准确率")
    report.append(
        f"- Baseline (阈值 {args.semantic_threshold}): {semantic_base_acc:.2f}% ({semantic_base_correct}/{total})"
    )
    report.append(
        f"- Memory (阈值 {args.semantic_threshold}): {semantic_mem_acc:.2f}% ({semantic_mem_correct}/{total})"
    )
    report.append("")
    report.append("## 检索命中率")
    report.append(f"- Evidence hit@{args.top_k}: {evidence_hit_rate:.2f}%")
    report.append("")
    report.append("## 记忆库大小")
    report.append(f"- Turn 数量: {size_stats['turns']}")
    report.append(f"- 文本字节数: {size_stats['text_bytes'] / (1024**2):.2f} MB")
    report.append(f"- Embedding 字节数: {size_stats['embedding_bytes'] / (1024**2):.2f} MB")
    report.append(f"- 总大小: {size_stats['total_bytes'] / (1024**2):.2f} MB")
    report.append("")

    (out_dir / "report.md").write_text("\n".join(report), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Outputs: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
