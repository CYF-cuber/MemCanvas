#!/usr/bin/env python3
"""
MSR-VTT canvas format/scale variants evaluation.
Reuses extracted frames and cached text embeddings when available.
"""
import argparse
import glob
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


def find_latest_frames_root(fps: int, nframes: int) -> Path | None:
    candidates = sorted(glob.glob("/home/cyf/codex/msrvtt_experiment_*"), reverse=True)
    for c in candidates:
        p = Path(c) / f"frames_fps{fps}_n{nframes}"
        if p.exists():
            return p
    return None


def load_captions() -> Dict[str, List[str]]:
    with open(CAPTIONS_PKL, "rb") as f:
        raw = pickle.load(f)
    captions = {}
    for vid, caps in raw.items():
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


def make_canvas(frames: List[Image.Image], caption: str, scale: float) -> Image.Image:
    base_fw, base_fh = 320, 180
    fw = max(64, int(base_fw * scale))
    fh = max(48, int(base_fh * scale))
    cols, rows = 2, 2
    grid_w = cols * fw
    grid_h = rows * fh
    text_h = max(60, int(120 * scale))
    canvas = Image.new("RGB", (grid_w, grid_h + text_h), (255, 255, 255))

    for i in range(rows * cols):
        x = (i % cols) * fw
        y = (i // cols) * fh
        if i < len(frames):
            img = frames[i].resize((fw, fh))
            canvas.paste(img, (x, y))
        else:
            blank = Image.new("RGB", (fw, fh), (240, 240, 240))
            canvas.paste(blank, (x, y))

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", max(10, int(18 * scale)))
    except Exception:
        font = ImageFont.load_default()
    max_width = grid_w - 20
    lines = wrap_text(caption, max_width, draw, font)
    y_text = grid_h + 10
    line_height = max(12, int(22 * scale))
    for line in lines[:4]:
        draw.text((10, y_text), line, fill=(0, 0, 0), font=font)
        y_text += line_height
    return canvas


class CLIPEncoder:
    def __init__(self, model_name: str = CLIP_MODEL):
        self.processor = CLIPProcessor.from_pretrained(model_name, local_files_only=True)
        self.model = CLIPModel.from_pretrained(model_name, local_files_only=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.model.eval()

    def encode_images(self, images: List[Image.Image]) -> np.ndarray:
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            feats = self.model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy()

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

    cap_idx_by_vid = {vid: [] for vid in video_ids}
    for idx, vid in enumerate(caption_video_ids):
        if vid in cap_idx_by_vid:
            cap_idx_by_vid[vid].append(idx)

    recalls = {k: 0 for k in ks}
    total = len(video_ids)
    batch = 256
    for i in tqdm(range(0, total, batch), desc="Video->Text retrieval"):
        vb = v[i:i+batch]
        sims = vb @ t.T
        for j in range(sims.shape[0]):
            vid = video_ids[i + j]
            cap_idxs = cap_idx_by_vid.get(vid, [])
            if not cap_idxs:
                continue
            max_gt = sims[j, cap_idxs].max()
            rank = (sims[j] > max_gt).sum().item() + 1
            for k in ks:
                if rank <= k:
                    recalls[k] += 1
    return {f"R@{k}": recalls[k] / total * 100.0 for k in ks}


def parse_variant(s: str) -> Dict:
    # format like: webp-85-s0.75 or avif-50-s1.0 or png-s1.0
    parts = s.split("-")
    fmt = parts[0].lower()
    quality = None
    scale = 1.0
    for p in parts[1:]:
        if p.startswith("s"):
            scale = float(p[1:])
        else:
            try:
                quality = int(p)
            except ValueError:
                pass
    return {"name": s, "format": fmt, "quality": quality, "scale": scale}


def save_canvas(img: Image.Image, path: Path, fmt: str, quality: int | None) -> None:
    if fmt == "png":
        img.save(path, format="PNG")
    elif fmt == "webp":
        q = 85 if quality is None else quality
        img.save(path, format="WEBP", quality=q, method=6)
    elif fmt == "avif":
        q = 50 if quality is None else quality
        img.save(path, format="AVIF", quality=q)
    else:
        raise ValueError(f"Unsupported format: {fmt}")


def format_supported(fmt: str) -> bool:
    img = Image.new("RGB", (2, 2), (255, 255, 255))
    try:
        img.save(os.devnull, format=fmt.upper())
    except Exception:
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--frames-per-video", type=int, default=4)
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--output-root", default=OUTPUT_ROOT)
    parser.add_argument("--frames-root", default="")
    parser.add_argument("--variants", nargs="+", default=["webp-85-s1.0", "avif-50-s1.0", "webp-85-s0.75", "webp-85-s0.5"])
    args = parser.parse_args()

    out_dir = Path(args.output_root) / f"msrvtt_canvas_variants_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    captions = load_captions()
    video_ids = load_split(args.split)
    if args.max_videos:
        video_ids = video_ids[:args.max_videos]

    frames_root = Path(args.frames_root) if args.frames_root else find_latest_frames_root(args.fps, args.frames_per_video)
    if frames_root is None:
        frames_root = out_dir / f"frames_fps{args.fps}_n{args.frames_per_video}"
    frames_root.mkdir(parents=True, exist_ok=True)

    encoder = CLIPEncoder()

    # cache text embeddings
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(exist_ok=True)
    text_emb_path = cache_dir / "text_embs.npy"
    cap_vid_path = cache_dir / "cap_video_ids.json"
    vid_path = cache_dir / "video_ids.json"

    aligned_video_ids = [vid for vid in video_ids if (VIDEO_DIR / f"{vid}.mp4").exists()]

    all_caps = []
    cap_video_ids = []
    for vid in aligned_video_ids:
        for cap in captions.get(vid, []):
            all_caps.append(cap)
            cap_video_ids.append(vid)

    if text_emb_path.exists() and cap_vid_path.exists() and vid_path.exists():
        text_embs = np.load(text_emb_path)
        cap_video_ids = json.loads(cap_vid_path.read_text())
        aligned_video_ids = json.loads(vid_path.read_text())
    else:
        text_embs = encoder.encode_texts(all_caps)
        np.save(text_emb_path, text_embs)
        cap_vid_path.write_text(json.dumps(cap_video_ids), encoding="utf-8")
        vid_path.write_text(json.dumps(aligned_video_ids), encoding="utf-8")

    vid_to_idx = {vid: i for i, vid in enumerate(aligned_video_ids)}
    gt_indices = np.array([vid_to_idx[vid] for vid in cap_video_ids], dtype=np.int64)

    # sizes: raw videos + raw captions + frames bytes
    raw_video_bytes = sum((VIDEO_DIR / f"{vid}.mp4").stat().st_size for vid in aligned_video_ids)
    raw_caption_bytes = sum(len(c.encode("utf-8")) for c in all_caps)
    frames_bytes = 0
    for vid in aligned_video_ids:
        frame_dir = frames_root / vid
        for fp in frame_dir.glob("frame_*.jpg"):
            frames_bytes += fp.stat().st_size

    results = []

    for variant in args.variants:
        cfg = parse_variant(variant)
        fmt = cfg["format"]
        if fmt == "avif" and not format_supported("AVIF"):
            print(f"Skip {variant}: AVIF not supported by PIL")
            continue
        if fmt == "webp" and not format_supported("WEBP"):
            print(f"Skip {variant}: WEBP not supported by PIL")
            continue

        canvas_dir = out_dir / f"canvas_{variant}"
        canvas_dir.mkdir(parents=True, exist_ok=True)

        video_embs = []
        canvas_bytes = 0
        batch_imgs = []
        batch_indices = []

        for vid in tqdm(aligned_video_ids, desc=f"Canvas {variant}"):
            video_path = VIDEO_DIR / f"{vid}.mp4"
            frame_dir = frames_root / vid
            if not frame_dir.exists():
                frame_paths = extract_frames(video_path, frame_dir, args.fps, args.frames_per_video)
            else:
                frame_paths = sorted(frame_dir.glob("frame_*.jpg"))[:args.frames_per_video]

            imgs = []
            for fp in frame_paths:
                try:
                    imgs.append(Image.open(fp).convert("RGB"))
                except Exception:
                    pass
            if not imgs:
                continue
            cap_list = captions.get(vid, [])
            caption = cap_list[0] if cap_list else ""
            canvas = make_canvas(imgs[:4], caption, cfg["scale"])
            canvas_path = canvas_dir / f"{vid}.{fmt}"
            save_canvas(canvas, canvas_path, fmt, cfg["quality"])
            canvas_bytes += canvas_path.stat().st_size

            batch_imgs.append(canvas)
            batch_indices.append(vid)
            if len(batch_imgs) >= 64:
                emb = encoder.encode_images(batch_imgs)
                video_embs.extend(list(emb))
                batch_imgs = []
                batch_indices = []

        if batch_imgs:
            emb = encoder.encode_images(batch_imgs)
            video_embs.extend(list(emb))

        video_embs = np.stack(video_embs, axis=0)

        metrics_t2v = compute_text_to_video_metrics(text_embs, video_embs, gt_indices)
        metrics_v2t = compute_video_to_text_metrics(text_embs, video_embs, cap_video_ids, aligned_video_ids)

        results.append({
            "variant": variant,
            "format": fmt,
            "quality": cfg["quality"],
            "scale": cfg["scale"],
            "text_to_video": metrics_t2v,
            "video_to_text": metrics_v2t,
            "sizes": {
                "raw_videos_bytes": raw_video_bytes,
                "raw_captions_bytes": raw_caption_bytes,
                "frames_bytes": frames_bytes,
                "canvas_bytes": canvas_bytes,
            },
        })

    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps({"split": args.split, "results": results}, indent=2), encoding="utf-8")

    report = []
    report.append("# MSR-VTT Canvas Variants 实验报告")
    report.append("")
    report.append(f"- Split: {args.split}")
    report.append(f"- Videos: {len(aligned_video_ids)}")
    report.append(f"- Captions: {len(all_caps)}")
    report.append(f"- Frames/video: {args.frames_per_video}, FPS: {args.fps}")
    report.append("")
    for r in results:
        report.append(f"## {r['variant']}")
        report.append(f"- Text→Video: {r['text_to_video']}")
        report.append(f"- Video→Text: {r['video_to_text']}")
        report.append(f"- Raw videos: {r['sizes']['raw_videos_bytes'] / (1024**3):.2f} GB")
        report.append(f"- Raw captions: {r['sizes']['raw_captions_bytes'] / (1024**2):.2f} MB")
        report.append(f"- Keyframes (JPEG): {r['sizes']['frames_bytes'] / (1024**2):.2f} MB")
        report.append(f"- Canvas: {r['sizes']['canvas_bytes'] / (1024**2):.2f} MB")
        report.append("")

    (out_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
