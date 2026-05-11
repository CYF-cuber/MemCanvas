"""
VisionTokenExtractor - Vision Token提取器

从切分后的画布patches中提取vision tokens。
支持多种vision encoder后端。

增强版支持：
- DeepSeek风格的Aligner投影
- Token压缩（resampler/pooling/conv）
- 位置编码
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union, Tuple, Any, Literal
from pathlib import Path
from PIL import Image
import numpy as np

from .canvas_slicer import SliceResult


@dataclass
class TokenExtractionConfig:
    """Token提取配置"""
    # 使用的vision encoder类型
    encoder_type: str = "clip"  # clip, siglip, clip-aligned
    # 模型名称/路径
    model_name: str = "openai/clip-vit-base-patch32"
    # 设备
    device: str = "cuda"
    # 是否返回attention weights
    return_attention: bool = False
    # batch size
    batch_size: int = 4

    # === DeepSeek风格Aligner配置 ===
    # 是否使用Aligner投影（模仿DeepSeek-VL）
    use_aligner: bool = False
    # Aligner输出维度
    aligner_output_dim: int = 1024
    # Aligner隐藏层维度
    aligner_hidden_dim: Optional[int] = None
    # 是否添加位置编码
    add_positional_encoding: bool = False

    # === Token压缩配置 ===
    # 压缩模式: none, pooling, resampler, conv
    compress_mode: str = "none"
    # 目标token数（用于resampler）
    target_tokens: int = 64
    # pooling模式: mean, max, cls
    pooling_mode: str = "mean"


@dataclass
class VisionTokens:
    """Vision Token结果"""
    # 全局视图的tokens [num_tokens, hidden_dim]
    global_tokens: np.ndarray
    # 各patch的tokens list of [num_tokens, hidden_dim]
    patch_tokens: List[np.ndarray]
    # 合并后的所有tokens [total_tokens, hidden_dim]
    all_tokens: np.ndarray
    # token数量统计
    num_global_tokens: int
    num_patch_tokens: int
    total_tokens: int
    # hidden dimension
    hidden_dim: int
    # key embedding用于检索 [hidden_dim]
    key_embedding: Optional[np.ndarray] = None
    # 是否经过了Aligner处理
    is_aligned: bool = False
    # 是否经过了压缩
    is_compressed: bool = False


class VisionTokenExtractor:
    """
    Vision Token提取器

    从SliceResult中提取vision tokens。

    支持两种模式：
    1. 基础模式：直接使用CLIP/SigLIP提取tokens
    2. DeepSeek风格模式：CLIP + Aligner投影 + 可选压缩
    """

    def __init__(self, config: Optional[TokenExtractionConfig] = None):
        self.config = config or TokenExtractionConfig()
        self.model = None
        self.processor = None
        self.aligner = None
        self.compressor = None
        self._initialized = False

    def _init_model(self):
        """延迟初始化模型"""
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
        """初始化CLIP模型"""
        try:
            from transformers import CLIPModel, CLIPProcessor
            self.model = CLIPModel.from_pretrained(self.config.model_name)
            self.processor = CLIPProcessor.from_pretrained(self.config.model_name)
            self.model.to(self.config.device)
            self.model.eval()

            # 如果启用aligner，初始化它
            if self.config.use_aligner:
                self._init_aligner()

        except ImportError:
            raise ImportError("请安装transformers: pip install transformers")

    def _init_siglip(self):
        """初始化SigLIP模型"""
        try:
            from transformers import SiglipModel, SiglipProcessor
            self.model = SiglipModel.from_pretrained(self.config.model_name)
            self.processor = SiglipProcessor.from_pretrained(self.config.model_name)
            self.model.to(self.config.device)
            self.model.eval()

            if self.config.use_aligner:
                self._init_aligner()

        except ImportError:
            raise ImportError("请安装transformers: pip install transformers")

    def _init_clip_aligned(self):
        """初始化CLIP + Aligner模式"""
        self._init_clip()
        if not self.config.use_aligner:
            self.config.use_aligner = True
            self._init_aligner()

    def _init_aligner(self):
        """初始化Aligner和Compressor"""
        import torch
        from .clip_aligner import (
            CLIPAligner, TokenCompressor,
            AlignerConfig, CompressorConfig
        )

        # 获取CLIP的hidden dimension
        if hasattr(self.model, 'config'):
            input_dim = self.model.config.vision_config.hidden_size
        else:
            input_dim = 768  # default for ViT-B

        # 创建Aligner配置
        aligner_config = AlignerConfig(
            input_dim=input_dim,
            output_dim=self.config.aligner_output_dim,
            hidden_dim=self.config.aligner_hidden_dim or self.config.aligner_output_dim,
            activation="gelu",
            use_layer_norm=True,
            add_positional_encoding=self.config.add_positional_encoding
        )

        # 创建Compressor配置
        compressor_config = CompressorConfig(
            compress_mode=self.config.compress_mode,
            target_tokens=self.config.target_tokens,
            pooling_mode=self.config.pooling_mode
        )

        # 初始化模块
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
        """从SliceResult提取vision tokens"""
        self._init_model()

        # 提取全局视图tokens
        global_tokens = self._extract_single(slice_result.global_view)

        # 提取patch tokens
        patch_tokens = []
        for patch in slice_result.patches:
            tokens = self._extract_single(patch)
            patch_tokens.append(tokens)

        # 合并所有tokens
        all_list = [global_tokens] + patch_tokens
        all_tokens = np.concatenate(all_list, axis=0)

        # 计算key embedding（用于检索）
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
        批量提取vision tokens

        Args:
            images: PIL.Image列表
        Returns:
            tokens列表，每个元素是 [seq_len, hidden_dim]
        """
        self._init_model()
        import torch

        results = []
        batch_size = self.config.batch_size

        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]

            # 预处理
            batch_images = [img.convert('RGB') if img.mode != 'RGB' else img for img in batch_images]
            inputs = self.processor(images=batch_images, return_tensors="pt")
            inputs = {k: v.to(self.config.device) for k, v in inputs.items()}

            with torch.no_grad():
                # CLIP提取
                outputs = self.model.vision_model(**inputs)
                hidden_states = outputs.last_hidden_state  # [batch, seq, hidden]

                # Aligner投影
                if self.aligner is not None:
                    hidden_states = self.aligner(hidden_states)

                # 压缩
                if self.compressor is not None:
                    hidden_states = self.compressor(hidden_states)

            # 转换为numpy
            batch_tokens = hidden_states.cpu().numpy()
            for j in range(batch_tokens.shape[0]):
                results.append(batch_tokens[j])

        return results

    def _extract_single(self, image: Image.Image) -> np.ndarray:
        """提取单张图像的vision tokens"""
        import torch

        if image.mode != 'RGB':
            image = image.convert('RGB')

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}

        with torch.no_grad():
            # 提取CLIP特征
            if self.config.encoder_type in ["clip", "clip-aligned"]:
                outputs = self.model.vision_model(**inputs)
            else:
                outputs = self.model.vision_model(**inputs)

            # 获取最后一层hidden states
            hidden_states = outputs.last_hidden_state  # [1, seq, hidden]

            # Aligner投影（DeepSeek风格）
            if self.aligner is not None:
                hidden_states = self.aligner(hidden_states)

            # Token压缩
            if self.compressor is not None:
                hidden_states = self.compressor(hidden_states)

        return hidden_states.cpu().numpy().squeeze(0)

    def get_config_summary(self) -> dict:
        """获取配置摘要"""
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
    创建DeepSeek风格的Vision Token提取器

    模仿DeepSeek-VL的设计：
    1. CLIP ViT提取patch tokens
    2. MLP Aligner投影到LLM空间
    3. 可选的token压缩

    Args:
        output_dim: 输出维度（通常匹配LLM的hidden_size）
        compress_mode: 压缩模式 (none, pooling, resampler, conv)
        target_tokens: 目标token数（用于resampler）
        model_name: CLIP模型名称
        device: 设备

    Returns:
        配置好的VisionTokenExtractor
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
