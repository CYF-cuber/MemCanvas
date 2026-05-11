#!/usr/bin/env python3
"""
MSR-VTT retrieval experiment with keyframes.
- Baseline: average CLIP image embeddings of keyframes
- Memory Canvas: a single canvas image (grid + caption text) per video
Evaluate text->video and video->text retrieval on test split.
"""
import argparse
import json
import os
import pickle
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

import torch
from transformers import CLIPProcessor, CLIPModel

MSRVTT_ROOT = Path("/home/cyf/codex/datasets/msrvtt/MSRVTT")
CAPTIONS_PKL = MSRVTT_ROOT / "high-quality/structured-symlinks/raw-captions.pkl"
SPLIT_DIR = MSRVTT_ROOT / "high-quality/structured-symlinks"
VIDEO_DIR = MSRVTT_ROOT / "videos/all"
CLIP_MODEL = "openai/clip-vit-large-patch14"
OUTPUT_ROOT = "/home/cyf/codex"


def load_captions() -> Dict[str, List[str]]:
    with open(CAPTIONS_PKL, "rb") as f:
        raw = pickle.load(f)
    captions = {}
    for vid, caps in raw.items():
        # caps: list of token lists
        captions[vid] = [" ".join(tokens) for tokens in caps]
    return captions


def load_split(name: str) -> List[str]:
    p = SPLIT_DIR / f"{name}_list_full.txt"
    lines = p.read_text().strip().splitlines()
    return [l.strip() for l in lines if l.strip()]


def extract_frames(video_path: Path, out_dir: Path, fps: int, max_frames: int) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("frame_*.jpg"))
    if len(existing) >= max_frames:
        return existing[:max_frames]

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-vframes", str(max_frames),
        str(out_dir / "frame_%05d.jpg"),
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return sorted(out_dir.glob("frame_*.jpg"))[:max_frames]


def wrap_text(text: str, max_width: int, draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont) -> List[str]:
    words = text.split()
    lines = []
    cur = []
    for w in words:
        test = " ".join(cur + [w])
        w_width = draw.textlength(test, font=font)
        if w_width <= max_width or not cur:
            cur.append(w)
        else:
            lines.append(" ".join(cur))
            cur = [w]
    if cur:
        lines.append(" ".join(cur))
    return lines


def make_canvas(frames: List[Image.Image], caption: str, frame_size=(320, 180), grid=(2, 2)) -> Image.Image:
    cols, rows = grid
    fw, fh = frame_size
    grid_w = cols * fw
    grid_h = rows * fh
    text_h = 120
    canvas = Image.new("RGB", (grid_w, grid_h + text_h), (255, 255, 255))

    # paste frames
    for i in range(rows * cols):
        x = (i % cols) * fw
        y = (i // cols) * fh
        if i < len(frames):
            img = frames[i].resize((fw, fh))
            canvas.paste(img, (x, y))
        else:
            blank = Image.new("RGB", (fw, fh), (240, 240, 240))
            canvas.paste(blank, (x, y))

    # caption text
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    max_width = grid_w - 20
    lines = wrap_text(caption, max_width, draw, font)
    y_text = grid_h + 10
    for line in lines[:4]:
        draw.text((10, y_text), line, fill=(0, 0, 0), font=font)
        y_text += 22
    return canvas


class CLIPEncoder:
    def __init__(self, model_name: str = CLIP_MODEL):
        # Force offline load to avoid transient SSL/network issues.
        self.processor = CLIPProcessor.from_pretrained(model_name, local_files_only=True)
        self.model = CLIPModel.from_pretrained(model_name, local_files_only=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.model.eval()

    def encode_images(self, images: List[Image.Image], batch_size: int = 32) -> np.ndarray:
        all_embeds = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            inputs = self.processor(images=batch, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                feats = self.model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_embeds.append(feats.cpu().numpy())
        return np.concatenate(all_embeds, axis=0) if all_embeds else np.zeros((0, 768), dtype=np.float32)

    def encode_texts(self, texts: List[str], batch_size: int = 128) -> np.ndarray:
        all_embeds = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encode captions"):
            batch = texts[i:i+batch_size]
            inputs = self.processor(text=batch, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                feats = self.model.get_text_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_embeds.append(feats.cpu().numpy())
        return np.concatenate(all_embeds, axis=0) if all_embeds else np.zeros((0, 768), dtype=np.float32)


def compute_text_to_video_metrics(text_embs: np.ndarray, video_embs: np.ndarray, gt_indices: np.ndarray, ks=(1,5,10)) -> Dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    v = torch.tensor(video_embs, device=device)
    t = torch.tensor(text_embs, device=device)

    recalls = {k: 0 for k in ks}
    total = t.shape[0]
    batch = 512
    for i in tqdm(range(0, total, batch), desc="Text->Video retrieval"):
        tb = t[i:i+batch]
        sims = tb @ v.T
        gt = torch.tensor(gt_indices[i:i+batch], device=device)
        gt_sims = sims[torch.arange(sims.shape[0], device=device), gt]
        ranks = (sims > gt_sims.unsqueeze(1)).sum(dim=1) + 1
        for k in ks:
            recalls[k] += (ranks <= k).sum().item()
    return {f"R@{k}": recalls[k] / total * 100.0 for k in ks}


def compute_video_to_text_metrics(text_embs: np.ndarray, video_embs: np.ndarray, caption_video_ids: List[str], video_ids: List[str], ks=(1,5,10)) -> Dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    v = torch.tensor(video_embs, device=device)
    t = torch.tensor(text_embs, device=device)

    # build caption indices per video
    cap_idx_by_vid = {vid: [] for vid in video_ids}
    for idx, vid in enumerate(caption_video_ids):
        if vid in cap_idx_by_vid:
            cap_idx_by_vid[vid].append(idx)

    recalls = {k: 0 for k in ks}
    total = len(video_ids)
    batch = 256
    for i in tqdm(range(0, total, batch), desc="Video->Text retrieval"):
        vb = v[i:i+batch]
        sims = vb @ t.T  # (b, num_caps)
        for j in range(sims.shape[0]):
            vid = video_ids[i + j]
            cap_idxs = cap_idx_by_vid.get(vid, [])
            if not cap_idxs:
                continue
            gt_sims = sims[j, cap_idxs]
            # best rank among correct captions
            # rank of a caption = 1 + count of captions with higher sim
            max_gt = gt_sims.max()
            rank = (sims[j] > max_gt).sum().item() + 1
            for k in ks:
                if rank <= k:
                    recalls[k] += 1
    return {f"R@{k}": recalls[k] / total * 100.0 for k in ks}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--frames-per-video", type=int, default=4)
    parser.add_argument("--max-videos", type=int, default=0, help="0 means all")
    parser.add_argument("--output-root", default=OUTPUT_ROOT)
    parser.add_argument("--frames-cache", type=str, default="",
                        help="Reuse frames from a previous run dir (e.g. .../frames_fps1_n4)")
    args = parser.parse_args()

    out_dir = Path(args.output_root) / f"msrvtt_experiment_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    captions = load_captions()
    video_ids = load_split(args.split)
    if args.max_videos:
        video_ids = video_ids[:args.max_videos]

    encoder = CLIPEncoder()

    frames_root = Path(args.frames_cache) if args.frames_cache else out_dir / f"frames_fps{args.fps}_n{args.frames_per_video}"
    canvas_root = out_dir / "canvas"
    canvas_root.mkdir(parents=True, exist_ok=True)

    video_emb_baseline = []
    video_emb_canvas = []

    frames_bytes = 0
    canvas_bytes = 0

    for vid in tqdm(video_ids, desc="Video embeddings"):
        video_path = VIDEO_DIR / f"{vid}.mp4"
        if not video_path.exists():
            continue
        frame_dir = frames_root / vid
        frame_paths = extract_frames(video_path, frame_dir, args.fps, args.frames_per_video)

        imgs = []
        for fp in frame_paths:
            try:
                imgs.append(Image.open(fp).convert("RGB"))
                frames_bytes += fp.stat().st_size
            except Exception:
                pass
        if not imgs:
            continue

        emb_frames = encoder.encode_images(imgs)
        emb_base = emb_frames.mean(axis=0)
        emb_base = emb_base / np.linalg.norm(emb_base)
        video_emb_baseline.append(emb_base)

        cap_list = captions.get(vid, [])
        caption = cap_list[0] if cap_list else ""
        canvas = make_canvas(imgs[:4], caption)
        canvas_path = canvas_root / f"{vid}.png"
        canvas.save(canvas_path)
        canvas_bytes += canvas_path.stat().st_size
        emb_canvas = encoder.encode_images([canvas])[0]
        video_emb_canvas.append(emb_canvas)

    # align video ids with embeddings (some videos may be missing)
    aligned_video_ids = [vid for vid in video_ids if (VIDEO_DIR / f"{vid}.mp4").exists()]
    video_emb_baseline = np.stack(video_emb_baseline, axis=0)
    video_emb_canvas = np.stack(video_emb_canvas, axis=0)

    # captions and text embeddings (for aligned videos only)
    all_caps = []
    cap_video_ids = []
    for vid in aligned_video_ids:
        for cap in captions.get(vid, []):
            all_caps.append(cap)
            cap_video_ids.append(vid)

    text_embs = encoder.encode_texts(all_caps)

    # map caption gt to video index
    vid_to_idx = {vid: i for i, vid in enumerate(aligned_video_ids)}
    gt_indices = np.array([vid_to_idx[vid] for vid in cap_video_ids], dtype=np.int64)

    metrics_baseline_t2v = compute_text_to_video_metrics(text_embs, video_emb_baseline, gt_indices)
    metrics_canvas_t2v = compute_text_to_video_metrics(text_embs, video_emb_canvas, gt_indices)
    metrics_baseline_v2t = compute_video_to_text_metrics(text_embs, video_emb_baseline, cap_video_ids, aligned_video_ids)
    metrics_canvas_v2t = compute_video_to_text_metrics(text_embs, video_emb_canvas, cap_video_ids, aligned_video_ids)

    # sizes
    raw_video_bytes = 0
    for vid in aligned_video_ids:
        raw_video_bytes += (VIDEO_DIR / f"{vid}.mp4").stat().st_size
    raw_caption_bytes = sum(len(c.encode("utf-8")) for c in all_caps)

    size_stats = {
        "raw_videos_bytes": raw_video_bytes,
        "raw_captions_bytes": raw_caption_bytes,
        "baseline_frames_bytes": frames_bytes,
        "canvas_bytes": canvas_bytes,
        "video_embedding_bytes": int(video_emb_baseline.nbytes),
    }

    summary = {
        "split": args.split,
        "videos": len(aligned_video_ids),
        "captions": len(all_caps),
        "frames_per_video": args.frames_per_video,
        "fps": args.fps,
        "baseline_text_to_video": metrics_baseline_t2v,
        "canvas_text_to_video": metrics_canvas_t2v,
        "baseline_video_to_text": metrics_baseline_v2t,
        "canvas_video_to_text": metrics_canvas_v2t,
        "sizes": size_stats,
    }

    (out_dir / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = []
    report.append("# MSR-VTT 实验报告")
    report.append("")
    report.append(f"- Split: {args.split}")
    report.append(f"- Videos: {len(aligned_video_ids)}")
    report.append(f"- Captions: {len(all_caps)}")
    report.append(f"- Frames/video: {args.frames_per_video}, FPS: {args.fps}")
    report.append("")
    report.append("## Text→Video Retrieval")
    report.append(f"- Baseline (frames avg): {metrics_baseline_t2v}")
    report.append(f"- Memory Canvas: {metrics_canvas_t2v}")
    report.append("")
    report.append("## Video→Text Retrieval")
    report.append(f"- Baseline (frames avg): {metrics_baseline_v2t}")
    report.append(f"- Memory Canvas: {metrics_canvas_v2t}")
    report.append("")
    report.append("## 存储大小")
    report.append(f"- Raw videos: {raw_video_bytes / (1024**3):.2f} GB")
    report.append(f"- Raw captions: {raw_caption_bytes / (1024**2):.2f} MB")
    report.append(f"- Keyframes (JPEG): {frames_bytes / (1024**2):.2f} MB")
    report.append(f"- Canvas images (PNG): {canvas_bytes / (1024**2):.2f} MB")
    report.append(f"- Video embeddings: {video_emb_baseline.nbytes / (1024**2):.2f} MB")

    (out_dir / "report.md").write_text("\n".join(report), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Outputs: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
