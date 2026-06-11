"""
VisionTokenExtractor - Vision Tokentext

textpatchestextvision tokens。
textvision encodertext。

text：
- DeepSeektextAlignertext
- Tokentext（resampler/pooling/conv）
- text
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union, Tuple, Any, Literal
from pathlib import Path
from PIL import Image
import numpy as np

from .canvas_slicer import SliceResult


@dataclass
class TokenExtractionConfig:
    """Tokentext"""
    # textvision encodertext
    encoder_type: str = "clip"  # clip, siglip, clip-aligned
    # text/text
    model_name: str = "openai/clip-vit-base-patch32"
    # text
    device: str = "cuda"
    # textattention weights
    return_attention: bool = False
    # batch size
    batch_size: int = 4

    # === DeepSeektextAlignertext ===
    # textAlignertext（textDeepSeek-VL）
    use_aligner: bool = False
    # Alignertext
    aligner_output_dim: int = 1024
    # Alignertext
    aligner_hidden_dim: Optional[int] = None
    # text
    add_positional_encoding: bool = False

    # === Tokentext ===
    # text: none, pooling, resampler, conv
    compress_mode: str = "none"
    # texttokentext（textresampler）
    target_tokens: int = 64
    # poolingtext: mean, max, cls
    pooling_mode: str = "mean"


@dataclass
class VisionTokens:
    """Vision Tokentext"""
    # texttokens [num_tokens, hidden_dim]
    global_tokens: np.ndarray
    # textpatchtexttokens list of [num_tokens, hidden_dim]
    patch_tokens: List[np.ndarray]
    # texttokens [total_tokens, hidden_dim]
    all_tokens: np.ndarray
    # tokentext
    num_global_tokens: int
    num_patch_tokens: int
    total_tokens: int
    # hidden dimension
    hidden_dim: int
    # key embeddingtext [hidden_dim]
    key_embedding: Optional[np.ndarray] = None
    # textAlignertext
    is_aligned: bool = False
    # text
    is_compressed: bool = False


class VisionTokenExtractor:
    """
    Vision Tokentext

    textSliceResulttextvision tokens。

    text：
    1. text：textCLIP/SigLIPtexttokens
    2. DeepSeektext：CLIP + Alignertext + text
    """

    def __init__(self, config: Optional[TokenExtractionConfig] = None):
        self.config = config or TokenExtractionConfig()
        self.model = None
        self.processor = None
        self.aligner = None
        self.compressor = None
        self._initialized = False

    def _init_model(self):
        """text"""
        if self._initialized:
            return

        if self.config.encoder_type == "clip":
            self._init_clip()
        elif self.config.encoder_type == "siglip":
            self._init_siglip()
        elif self.config.encoder_type == "clip-aligned":
            self._init_clip_aligned()
        else:
            raise ValueError(f"Unknown encoder: {self.config.encoder_type}")

        self._initialized = True

    def _init_clip(self):
        """textCLIPtext"""
        try:
            from transformers import CLIPModel, CLIPProcessor
            self.model = CLIPModel.from_pretrained(self.config.model_name)
            self.processor = CLIPProcessor.from_pretrained(self.config.model_name)
            self.model.to(self.config.device)
            self.model.eval()

            # textaligner，text
            if self.config.use_aligner:
                self._init_aligner()

        except ImportError:
            raise ImportError("texttransformers: pip install transformers")

    def _init_siglip(self):
        """textSigLIPtext"""
        try:
            from transformers import SiglipModel, SiglipProcessor
            self.model = SiglipModel.from_pretrained(self.config.model_name)
            self.processor = SiglipProcessor.from_pretrained(self.config.model_name)
            self.model.to(self.config.device)
            self.model.eval()

            if self.config.use_aligner:
                self._init_aligner()

        except ImportError:
            raise ImportError("texttransformers: pip install transformers")

    def _init_clip_aligned(self):
        """textCLIP + Alignertext"""
        self._init_clip()
        if not self.config.use_aligner:
            self.config.use_aligner = True
            self._init_aligner()

    def _init_aligner(self):
        """textAlignertextCompressor"""
        import torch
        from .clip_aligner import (
            CLIPAligner, TokenCompressor,
            AlignerConfig, CompressorConfig
        )

        # textCLIPtexthidden dimension
        if hasattr(self.model, 'config'):
            input_dim = self.model.config.vision_config.hidden_size
        else:
            input_dim = 768  # default for ViT-B

        # textAlignertext
        aligner_config = AlignerConfig(
            input_dim=input_dim,
            output_dim=self.config.aligner_output_dim,
            hidden_dim=self.config.aligner_hidden_dim or self.config.aligner_output_dim,
            activation="gelu",
            use_layer_norm=True,
            add_positional_encoding=self.config.add_positional_encoding
        )

        # textCompressortext
        compressor_config = CompressorConfig(
            compress_mode=self.config.compress_mode,
            target_tokens=self.config.target_tokens,
            pooling_mode=self.config.pooling_mode
        )

        # text
        self.aligner = CLIPAligner(aligner_config)
        self.aligner.to(self.config.device)
        self.aligner.eval()

        self.compressor = TokenCompressor(
            compressor_config,
            self.config.aligner_output_dim
        )
        self.compressor.to(self.config.device)
        self.compressor.eval()

    def extract(self, slice_result: SliceResult) -> VisionTokens:
        """textSliceResulttextvision tokens"""
        self._init_model()

        # texttokens
        global_tokens = self._extract_single(slice_result.global_view)

        # textpatch tokens
        patch_tokens = []
        for patch in slice_result.patches:
            tokens = self._extract_single(patch)
            patch_tokens.append(tokens)

        # texttokens
        all_list = [global_tokens] + patch_tokens
        all_tokens = np.concatenate(all_list, axis=0)

        # textkey embedding（text）
        key_embedding = global_tokens.mean(axis=0)

        return VisionTokens(
            global_tokens=global_tokens,
            patch_tokens=patch_tokens,
            all_tokens=all_tokens,
            num_global_tokens=global_tokens.shape[0],
            num_patch_tokens=sum(t.shape[0] for t in patch_tokens),
            total_tokens=all_tokens.shape[0],
            hidden_dim=global_tokens.shape[-1],
            key_embedding=key_embedding,
            is_aligned=self.config.use_aligner,
            is_compressed=self.config.compress_mode != "none"
        )

    def extract_batch(self, images: List[Image.Image]) -> List[np.ndarray]:
        """
        textvision tokens

        Args:
            images: PIL.Imagetext
        Returns:
            tokenstext，text [seq_len, hidden_dim]
        """
        self._init_model()
        import torch

        results = []
        batch_size = self.config.batch_size

        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]

            # text
            batch_images = [img.convert('RGB') if img.mode != 'RGB' else img for img in batch_images]
            inputs = self.processor(images=batch_images, return_tensors="pt")
            inputs = {k: v.to(self.config.device) for k, v in inputs.items()}

            with torch.no_grad():
                # CLIPtext
                outputs = self.model.vision_model(**inputs)
                hidden_states = outputs.last_hidden_state  # [batch, seq, hidden]

                # Alignertext
                if self.aligner is not None:
                    hidden_states = self.aligner(hidden_states)

                # text
                if self.compressor is not None:
                    hidden_states = self.compressor(hidden_states)

            # textnumpy
            batch_tokens = hidden_states.cpu().numpy()
            for j in range(batch_tokens.shape[0]):
                results.append(batch_tokens[j])

        return results

    def _extract_single(self, image: Image.Image) -> np.ndarray:
        """textvision tokens"""
        import torch

        if image.mode != 'RGB':
            image = image.convert('RGB')

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}

        with torch.no_grad():
            # textCLIPtext
            if self.config.encoder_type in ["clip", "clip-aligned"]:
                outputs = self.model.vision_model(**inputs)
            else:
                outputs = self.model.vision_model(**inputs)

            # texthidden states
            hidden_states = outputs.last_hidden_state  # [1, seq, hidden]

            # Alignertext（DeepSeektext）
            if self.aligner is not None:
                hidden_states = self.aligner(hidden_states)

            # Tokentext
            if self.compressor is not None:
                hidden_states = self.compressor(hidden_states)

        return hidden_states.cpu().numpy().squeeze(0)

    def get_config_summary(self) -> dict:
        """text"""
        return {
            "encoder_type": self.config.encoder_type,
            "model_name": self.config.model_name,
            "use_aligner": self.config.use_aligner,
            "aligner_output_dim": self.config.aligner_output_dim if self.config.use_aligner else None,
            "compress_mode": self.config.compress_mode,
            "target_tokens": self.config.target_tokens if self.config.compress_mode == "resampler" else None
        }


def create_deepseek_style_extractor(
    output_dim: int = 1024,
    compress_mode: str = "none",
    target_tokens: int = 64,
    model_name: str = "openai/clip-vit-base-patch32",
    device: str = "cuda"
) -> VisionTokenExtractor:
    """
    textDeepSeektextVision Tokentext

    textDeepSeek-VLtext：
    1. CLIP ViTtextpatch tokens
    2. MLP AlignertextLLMtext
    3. texttokentext

    Args:
        output_dim: text（textLLMtexthidden_size）
        compress_mode: text (none, pooling, resampler, conv)
        target_tokens: texttokentext（textresampler）
        model_name: CLIPtext
        device: text

    Returns:
        textVisionTokenExtractor
    """
    config = TokenExtractionConfig(
        encoder_type="clip-aligned",
        model_name=model_name,
        device=device,
        use_aligner=True,
        aligner_output_dim=output_dim,
        add_positional_encoding=True,
        compress_mode=compress_mode,
        target_tokens=target_tokens
    )
    return VisionTokenExtractor(config)
