"""
Feature extraction for Layout Scorer.

Three feature groups:
1. CLIP visual features [768-dim] — from CLIP ViT-L/14 image encoder
2. Geometric features [8-dim] — from Layout object
3. Typographic features [7-dim] — from Layout placements
"""

import torch
import torch.nn.functional as F
from typing import List, Optional
from PIL import Image

from ..rendering.smart_canvas_layout import BlockType, Layout


# ---------------------------------------------------------------------------
# Geometric features [8-dim]
# ---------------------------------------------------------------------------

def extract_geometric_features(layout: Layout) -> List[float]:
    """Extract 8-dim geometric features from a Layout object."""
    return [
        layout.squareness,                                      # text [0,1]
        layout.utilization,                                     # text [0,1]
        layout.width / 1000.0,                                  # text
        layout.height / 1000.0,                                 # text
        layout.aspect_ratio,                                    # text
        (layout.width // 28) * (layout.height // 28) / 2000.0,  # vision token count (normalized)
        len(layout.placements) / 10.0,                          # block count (normalized)
        1.0 if 400 <= layout.width <= 900 and
              400 <= layout.height <= 900 else 0.0,             # size reasonableness
    ]


# ---------------------------------------------------------------------------
# Typographic features [7-dim]
# ---------------------------------------------------------------------------

def extract_typographic_features(layout: Layout) -> List[float]:
    """Extract 7-dim typographic features from a Layout object."""
    text_blocks = [p for p in layout.placements if p.block.type == BlockType.TEXT]
    image_blocks = [p for p in layout.placements if p.block.type == BlockType.IMAGE]
    table_blocks = [p for p in layout.placements if p.block.type == BlockType.TABLE]

    min_text_w = min((p.width for p in text_blocks), default=0)
    total_text_area = sum(p.width * p.height for p in text_blocks)
    total_image_area = sum(p.width * p.height for p in image_blocks)

    return [
        min_text_w / 600.0,                                     # narrowest text column (norm)
        1.0 if min_text_w >= 250 else min_text_w / 250.0,      # text readability
        total_text_area / max(layout.total_area, 1),            # text area ratio
        total_image_area / max(layout.total_area, 1),           # image area ratio
        1.0 if image_blocks else 0.0,                           # has image
        1.0 if table_blocks else 0.0,                           # has table
        len(text_blocks) / 6.0,                                 # text block count (norm)
    ]


# ---------------------------------------------------------------------------
# Combined handcrafted features [15-dim]
# ---------------------------------------------------------------------------

def extract_heuristic_features(layout: Layout) -> List[float]:
    """Extract all 15-dim handcrafted features (geometric + typographic)."""
    geo = extract_geometric_features(layout)
    typo = extract_typographic_features(layout)
    return geo + typo


def heuristic_score(layout: Layout) -> float:
    """Compute the original heuristic score used for baseline comparison."""
    sq = layout.squareness
    util = layout.utilization
    size_ok = 1.0 if 300 <= layout.width <= 1200 and 300 <= layout.height <= 1200 else 0.7
    return sq * 0.6 + util * 0.3 + size_ok * 0.1


# ---------------------------------------------------------------------------
# CLIP visual features [768-dim]
# ---------------------------------------------------------------------------

def load_clip_model(device: str = "cuda"):
    """Load CLIP ViT-L/14 model and processor."""
    from transformers import CLIPModel, CLIPProcessor

    model_name = "openai/clip-vit-large-patch14"
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()
    return model, processor


def extract_clip_features(
    image: Image.Image,
    clip_model,
    clip_processor,
    device: str = "cuda",
) -> torch.Tensor:
    """Extract CLIP visual features for a single image. Returns [768] tensor."""
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
    return F.normalize(features, dim=-1).squeeze(0).cpu()  # [768]


def batch_extract_clip(
    images: List[Image.Image],
    clip_model,
    clip_processor,
    device: str = "cuda",
    batch_size: int = 32,
) -> torch.Tensor:
    """Extract CLIP features for a batch of images. Returns [N, 768] tensor."""
    all_features = []

    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        inputs = clip_processor(images=batch, return_tensors="pt").to(device)
        with torch.no_grad():
            features = clip_model.get_image_features(**inputs)
        features = F.normalize(features, dim=-1).cpu()
        all_features.append(features)

    return torch.cat(all_features, dim=0)  # [N, 768]
