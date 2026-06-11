"""CLIP embedding and hybrid retrieval utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


def normalize_rows(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    denom = np.linalg.norm(array, axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return array / denom


def hybrid_keys(image_embeddings: np.ndarray, text_embeddings: np.ndarray, alpha: float = 0.75) -> np.ndarray:
    if image_embeddings.shape != text_embeddings.shape:
        raise ValueError("image_embeddings and text_embeddings must have the same shape")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    keys = alpha * image_embeddings + (1 - alpha) * text_embeddings
    return normalize_rows(keys)


def hybrid_retrieval(
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    alpha: float = 0.75,
    top_k: int = 2,
    threshold: float = 0.1,
) -> dict[int, list[tuple[int, float]]]:
    keys = hybrid_keys(image_embeddings, text_embeddings, alpha=alpha)
    queries = normalize_rows(query_embeddings)
    similarities = queries @ keys.T
    retrieval_map: dict[int, list[tuple[int, float]]] = {}
    for idx in range(len(queries)):
        ranked = np.argsort(similarities[idx])[::-1][: top_k + 10]
        retrieval_map[idx] = [
            (int(mem_idx), float(similarities[idx][mem_idx]))
            for mem_idx in ranked
            if similarities[idx][mem_idx] >= threshold
        ][:top_k]
    return retrieval_map


def load_clip(model_name: str = "openai/clip-vit-large-patch14"):
    from transformers import CLIPModel, CLIPProcessor

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor, device


def embed_images(
    images: Iterable[str | Path | Image.Image],
    model_name: str = "openai/clip-vit-large-patch14",
    batch_size: int = 32,
) -> np.ndarray:
    import torch

    model, processor, device = load_clip(model_name)
    image_list = [Image.open(img).convert("RGB") if isinstance(img, (str, Path)) else img.convert("RGB") for img in images]
    outputs = []
    for start in range(0, len(image_list), batch_size):
        batch = image_list[start : start + batch_size]
        inputs = processor(images=batch, return_tensors="pt", padding=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            features = model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        outputs.append(features.cpu().numpy())
    return np.concatenate(outputs, axis=0) if outputs else np.empty((0, 0), dtype=np.float32)


def embed_texts(
    texts: list[str],
    model_name: str = "openai/clip-vit-large-patch14",
    batch_size: int = 64,
    max_length: int = 77,
) -> np.ndarray:
    import torch

    model, processor, device = load_clip(model_name)
    outputs = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            features = model.get_text_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        outputs.append(features.cpu().numpy())
    return np.concatenate(outputs, axis=0) if outputs else np.empty((0, 0), dtype=np.float32)
