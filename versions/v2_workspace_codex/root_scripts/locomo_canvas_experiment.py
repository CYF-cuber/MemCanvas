#!/usr/bin/env python3
"""
LoCoMo-10 QA with Memory Canvas.
- Baseline: question only
- Memory: retrieve top-k memory canvas images (text rendered to image)
Outputs results and report under /home/cyf/codex.
"""
import argparse
import io
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from transformers import CLIPProcessor, CLIPModel
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from sentence_transformers import SentenceTransformer

import sys
sys.path.insert(0, "/home/cyf/memory")
from memory_canvas.dynamic_canvas import DynamicCanvas, DynamicCanvasConfig  # noqa: E402

DATA_PATH_DEFAULT = "/home/cyf/codex/datasets/locomo/locomo10.json"
MODEL_QWEN_DEFAULT = "/home/cyf/Qwen2.5-VL-7B-Instruct"
CLIP_MODEL = "openai/clip-vit-large-patch14"
SEMANTIC_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_ROOT = "/home/cyf/codex"


def get_local_semantic_model() -> str:
    base = Path("/home/cyf/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots")
    if base.exists():
        snaps = sorted(base.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if snaps:
            return str(snaps[0])
    return SEMANTIC_MODEL


@dataclass
class MemoryTurn:
    sample_id: str
    dia_id: str
    speaker: str
    text: str
    canvas_bytes: bytes
    embedding: np.ndarray


class CLIPEncoders:
    def __init__(self, model_name: str = CLIP_MODEL):
        # Force offline to avoid network calls.
        self.processor = CLIPProcessor.from_pretrained(model_name, local_files_only=True)
        self.model = CLIPModel.from_pretrained(model_name, local_files_only=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.model.eval()

    def encode_image(self, image) -> np.ndarray:
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            feats = self.model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().squeeze()

    def encode_text(self, text: str) -> np.ndarray:
        inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            feats = self.model.get_text_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().squeeze()


class QwenAnswerer:
    def __init__(self, model_path: str = MODEL_QWEN_DEFAULT):
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()

    def answer(self, question: str, memory_images: List) -> str:
        prompt_parts = []
        if memory_images:
            prompt_parts.append("Below are memory canvas images from past conversations.")
            prompt_parts.append("Each canvas contains: the speaker identity, their dialogue message, and session context.")
            prompt_parts.append("Study these canvases and use the conversation details to answer the question.")
            prompt_parts.append("---")
        prompt_parts.append("Answer the question briefly and directly.")
        prompt_parts.append(f"Question: {question}")
        prompt = "\n".join(prompt_parts)

        content = []
        for img in memory_images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if memory_images:
            inputs = self.processor(text=[text], images=memory_images, return_tensors="pt", padding=True)
        else:
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


def semantic_match_scores(preds: List[str], golds: List[str], thresholds: List[float]) -> Dict[float, Tuple[float, int]]:
    if not preds:
        return {t: (0.0, 0) for t in thresholds}
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    model_path = get_local_semantic_model()
    model = SentenceTransformer(
        model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        local_files_only=True,
    )
    pred_emb = model.encode(preds, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
    gold_emb = model.encode(golds, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
    sims = (pred_emb * gold_emb).sum(axis=1)
    out = {}
    for t in thresholds:
        correct = int((sims >= t).sum())
        acc = (correct / len(preds)) * 100.0
        out[t] = (acc, correct)
    return out


def parse_conversation(item: Dict) -> List[Dict]:
    conv = item.get("conversation", {})
    session_keys = [k for k in conv.keys() if k.startswith("session_") and not k.endswith("date_time")]

    def session_idx(k: str) -> int:
        try:
            return int(k.split("_")[1])
        except Exception:
            return 0

    session_keys.sort(key=session_idx)

    turns = []
    for sk in session_keys:
        for msg in conv.get(sk, []):
            turns.append(msg)
    return turns


def build_canvas_image(sample_id: str, msg: Dict) -> bytes:
    canvas = DynamicCanvas(DynamicCanvasConfig(
        patch_size=640,
        font_size=18,
        padding=24,
        content_gap=12,
        show_patch_boundary=False,
    ))
    dia_id = msg.get("dia_id", "")
    speaker = msg.get("speaker", "")
    text = msg.get("text", "")

    canvas.add_text(f"{sample_id} | {dia_id} | {speaker}", font_size=20)
    canvas.add_separator()
    canvas.add_text(text, font_size=18)

    patches = canvas.get_images()
    if len(patches) == 1:
        img = patches[0]
    else:
        total_height = sum(p.height for p in patches)
        max_width = max(p.width for p in patches)
        img = Image.new("RGB", (max_width, total_height), (255, 255, 255))
        y_offset = 0
        for p in patches:
            img.paste(p, (0, y_offset))
            y_offset += p.height

    out = io.BytesIO()
    img.convert("RGB").save(out, format="PNG")
    return out.getvalue()


def build_memory(data: List[Dict], encoder: CLIPEncoders) -> Dict[str, Dict]:
    mem = {}
    for item in data:
        sample_id = item.get("sample_id", "")
        turns = parse_conversation(item)
        entries: List[MemoryTurn] = []
        embeddings = []
        for msg in turns:
            canvas_bytes = build_canvas_image(sample_id, msg)
            img = Image.open(io.BytesIO(canvas_bytes)).convert("RGB")
            emb = encoder.encode_image(img)
            entries.append(MemoryTurn(
                sample_id=sample_id,
                dia_id=msg.get("dia_id", ""),
                speaker=msg.get("speaker", ""),
                text=msg.get("text", ""),
                canvas_bytes=canvas_bytes,
                embedding=emb,
            ))
            embeddings.append(emb)
        mem[sample_id] = {
            "entries": entries,
            "embeddings": np.stack(embeddings, axis=0) if embeddings else np.zeros((0, 768), dtype=np.float32),
        }
    return mem


def retrieve_top_k(question: str, mem_entry: Dict, encoder: CLIPEncoders, top_k: int) -> List[int]:
    q_emb = encoder.encode_text(question)
    if mem_entry["embeddings"].shape[0] == 0:
        return []
    sims = mem_entry["embeddings"] @ q_emb
    top_idx = np.argsort(-sims)[:top_k]
    return [int(i) for i in top_idx]


def compute_memory_size(mem: Dict[str, Dict]) -> Dict:
    canvas_bytes = 0
    embed_bytes = 0
    turns_count = 0
    for entry in mem.values():
        turns = entry["entries"]
        turns_count += len(turns)
        for t in turns:
            canvas_bytes += len(t.canvas_bytes)
        embed_bytes += entry["embeddings"].nbytes
    return {
        "turns": turns_count,
        "canvas_bytes": canvas_bytes,
        "embedding_bytes": embed_bytes,
        "total_bytes": canvas_bytes + embed_bytes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DATA_PATH_DEFAULT)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--output-root", default=OUTPUT_ROOT)
    parser.add_argument("--qwen-model", default=MODEL_QWEN_DEFAULT)
    parser.add_argument("--max-qa", type=int, default=0, help="0 means all")
    parser.add_argument("--semantic-thresholds", nargs="+", type=float, default=[0.5, 0.6, 0.7, 0.8])
    args = parser.parse_args()

    out_dir = Path(args.output_root) / f"locomo_canvas_experiment_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.data, "r", encoding="utf-8") as f:
        data = json.load(f)

    encoder = CLIPEncoders()
    mem = build_memory(data, encoder)
    size_stats = compute_memory_size(mem)

    # save sample canvases
    sample_dir = out_dir / "samples"
    sample_dir.mkdir(exist_ok=True)
    saved = 0
    for item in data:
        sample_id = item.get("sample_id", "")
        entry = mem[sample_id]
        for idx, m in enumerate(entry["entries"][:5]):
            if saved >= 10:
                break
            (sample_dir / f"{sample_id}_{idx}.png").write_bytes(m.canvas_bytes)
            saved += 1
        if saved >= 10:
            break

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
            retrieved_turns = [mem_entry["entries"][idx] for idx in top]
            if evidence:
                if any(t.dia_id in evidence for t in retrieved_turns):
                    evidence_hit += 1

            # baseline
            pred_base = answerer.answer(question, [])

            # memory canvas images
            memory_images = []
            for t in retrieved_turns:
                img = Image.open(io.BytesIO(t.canvas_bytes)).convert("RGB")
                memory_images.append(img)
            pred_mem = answerer.answer(question, memory_images)

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

    semantic = semantic_match_scores(base_preds, golds, args.semantic_thresholds)
    semantic_mem = semantic_match_scores(mem_preds, golds, args.semantic_thresholds)

    semantic_summary = {
        str(t): {
            "baseline_accuracy": semantic[t][0],
            "baseline_correct": semantic[t][1],
            "memory_accuracy": semantic_mem[t][0],
            "memory_correct": semantic_mem[t][1],
        } for t in args.semantic_thresholds
    }

    summary = {
        "total": total,
        "top_k": args.top_k,
        "baseline": {"correct": baseline_correct, "accuracy": baseline_acc},
        "memory": {"correct": memory_correct, "accuracy": memory_acc},
        "semantic": semantic_summary,
        "evidence_hit_rate": evidence_hit_rate,
        "memory_size": size_stats,
    }

    (out_dir / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "predictions.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    report = []
    report.append("# LoCoMo-10 Canvas 实验报告")
    report.append("")
    report.append(f"- QA 数量: {total}")
    report.append(f"- Top-K: {args.top_k}")
    report.append("")
    report.append("## 准确率")
    report.append(f"- Baseline (无记忆): {baseline_acc:.2f}% ({baseline_correct}/{total})")
    report.append(f"- Memory Canvas: {memory_acc:.2f}% ({memory_correct}/{total})")
    report.append("")
    report.append("## 语义匹配准确率")
    for t in args.semantic_thresholds:
        s = semantic_summary[str(t)]
        report.append(
            f"- 阈值 {t}: Baseline {s['baseline_accuracy']:.2f}% ({s['baseline_correct']}/{total}), "
            f"Memory {s['memory_accuracy']:.2f}% ({s['memory_correct']}/{total})"
        )
    report.append("")
    report.append("## 检索命中率")
    report.append(f"- Evidence hit@{args.top_k}: {evidence_hit_rate:.2f}%")
    report.append("")
    report.append("## 记忆库大小")
    report.append(f"- Turn 数量: {size_stats['turns']}")
    report.append(f"- Canvas 字节数: {size_stats['canvas_bytes'] / (1024**2):.2f} MB")
    report.append(f"- Embedding 字节数: {size_stats['embedding_bytes'] / (1024**2):.2f} MB")
    report.append(f"- 总大小: {size_stats['total_bytes'] / (1024**2):.2f} MB")
    report.append("")
    report.append("## 图例")
    report.append(f"- 示例画布: {sample_dir}")

    (out_dir / "report.md").write_text("\n".join(report), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Outputs: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
