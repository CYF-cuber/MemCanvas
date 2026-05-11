"""
CLIPAligner - CLIP Vision Token Aligner

模仿 DeepSeek-VL 的设计，对 CLIP 提取的 vision tokens 进行投影和压缩处理。

关键组件：
1. CLIPAligner: MLP投影层，将vision tokens投影到目标维度
2. TokenCompressor: 可选的token压缩模块
3. PositionalEncoding: 位置编码（可选）
"""

from dataclasses import dataclass
from typing import Optional, List, Literal, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AlignerConfig:
    """Aligner配置"""
    # 输入维度（来自CLIP）
    input_dim: int = 768
    # 输出维度（目标空间）
    output_dim: int = 1024
    # 隐藏层维度（None则使用output_dim）
    hidden_dim: Optional[int] = None
    # 激活函数
    activation: str = "gelu"
    # 是否使用LayerNorm
    use_layer_norm: bool = True
    # Dropout率
    dropout: float = 0.0
    # 是否添加位置编码
    add_positional_encoding: bool = False
    # 最大序列长度（用于位置编码）
    max_seq_len: int = 1024


@dataclass
class CompressorConfig:
    """Token压缩配置"""
    # 压缩方式: none, pooling, resampler, conv
    compress_mode: str = "none"
    # 目标token数量（用于resampler）
    target_tokens: int = 64
    # pooling模式: mean, max, cls
    pooling_mode: str = "mean"
    # 卷积压缩的kernel和stride
    conv_kernel: int = 2
    conv_stride: int = 2


class CLIPAligner(nn.Module):
    """
    CLIP Vision Token Aligner

    类似DeepSeek-VL的设计：Linear -> Activation -> Linear
    将CLIP的vision tokens投影到目标维度空间。
    """

    def __init__(self, config: AlignerConfig):
        super().__init__()
        self.config = config

        hidden_dim = config.hidden_dim or config.output_dim

        # Layer 1: input -> hidden
        self.linear1 = nn.Linear(config.input_dim, hidden_dim)

        # Activation
        if config.activation == "gelu":
            self.activation = nn.GELU()
        elif config.activation == "relu":
            self.activation = nn.ReLU()
        elif config.activation == "silu":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {config.activation}")

        # Layer 2: hidden -> output
        self.linear2 = nn.Linear(hidden_dim, config.output_dim)

        # Optional LayerNorm
        self.layer_norm = nn.LayerNorm(config.output_dim) if config.use_layer_norm else None

        # Dropout
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else None

        # Optional positional encoding
        self.positional_encoding = None
        if config.add_positional_encoding:
            self.positional_encoding = nn.Parameter(
                torch.zeros(1, config.max_seq_len, config.output_dim)
            )
            nn.init.trunc_normal_(self.positional_encoding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, input_dim] vision tokens from CLIP
        Returns:
            [batch, seq_len, output_dim] aligned tokens
        """
        # MLP projection
        x = self.linear1(x)
        x = self.activation(x)
        if self.dropout:
            x = self.dropout(x)
        x = self.linear2(x)

        # LayerNorm
        if self.layer_norm:
            x = self.layer_norm(x)

        # Add positional encoding
        if self.positional_encoding is not None:
            seq_len = x.size(1)
            x = x + self.positional_encoding[:, :seq_len, :]

        return x


class TokenCompressor(nn.Module):
    """
    Token压缩模块

    支持多种压缩策略：
    1. pooling: 简单的mean/max pooling
    2. resampler: 使用可学习的query进行cross-attention压缩
    3. conv: 使用1D卷积进行下采样
    """

    def __init__(self, config: CompressorConfig, hidden_dim: int):
        super().__init__()
        self.config = config
        self.hidden_dim = hidden_dim

        if config.compress_mode == "resampler":
            # 可学习的query tokens
            self.query_tokens = nn.Parameter(
                torch.zeros(1, config.target_tokens, hidden_dim)
            )
            nn.init.trunc_normal_(self.query_tokens, std=0.02)

            # Cross-attention
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=8,
                batch_first=True
            )
            self.norm = nn.LayerNorm(hidden_dim)

        elif config.compress_mode == "conv":
            # 1D卷积下采样
            self.conv = nn.Conv1d(
                hidden_dim, hidden_dim,
                kernel_size=config.conv_kernel,
                stride=config.conv_stride
            )
            self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, hidden_dim]
        Returns:
            compressed tokens
        """
        if self.config.compress_mode == "none":
            return x

        elif self.config.compress_mode == "pooling":
            return self._pooling_compress(x)

        elif self.config.compress_mode == "resampler":
            return self._resampler_compress(x)

        elif self.config.compress_mode == "conv":
            return self._conv_compress(x)

        return x

    def _pooling_compress(self, x: torch.Tensor) -> torch.Tensor:
        """简单pooling压缩"""
        if self.config.pooling_mode == "mean":
            return x.mean(dim=1, keepdim=True)
        elif self.config.pooling_mode == "max":
            return x.max(dim=1, keepdim=True)[0]
        elif self.config.pooling_mode == "cls":
            # 返回第一个token（假设是CLS token）
            return x[:, :1, :]
        return x

    def _resampler_compress(self, x: torch.Tensor) -> torch.Tensor:
        """使用可学习query的cross-attention压缩"""
        batch_size = x.size(0)
        queries = self.query_tokens.expand(batch_size, -1, -1)

        # Cross-attention: query attends to vision tokens
        compressed, _ = self.cross_attn(queries, x, x)
        compressed = self.norm(compressed + queries)

        return compressed

    def _conv_compress(self, x: torch.Tensor) -> torch.Tensor:
        """1D卷积下采样"""
        # [batch, seq, dim] -> [batch, dim, seq]
        x = x.transpose(1, 2)
        x = self.conv(x)
        # [batch, dim, seq'] -> [batch, seq', dim]
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x


class CLIPVisionProcessor(nn.Module):
    """
    完整的CLIP Vision处理流水线

    流程：
    1. CLIP ViT 提取 patch tokens
    2. CLIPAligner 投影到目标维度
    3. TokenCompressor 可选压缩
    4. 输出处理后的 vision tokens
    """

    def __init__(
        self,
        aligner_config: Optional[AlignerConfig] = None,
        compressor_config: Optional[CompressorConfig] = None,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        device: str = "cuda"
    ):
        super().__init__()
        self.device = device
        self.clip_model_name = clip_model_name

        # 默认配置
        self.aligner_config = aligner_config or AlignerConfig()
        self.compressor_config = compressor_config or CompressorConfig()

        # CLIP模型（延迟加载）
        self.clip_model = None
        self.clip_processor = None

        # Aligner
        self.aligner = CLIPAligner(self.aligner_config)

        # Compressor
        self.compressor = TokenCompressor(
            self.compressor_config,
            self.aligner_config.output_dim
        )

        self._initialized = False

    def _init_clip(self):
        """延迟初始化CLIP"""
        if self._initialized:
            return

        from transformers import CLIPModel, CLIPProcessor

        self.clip_model = CLIPModel.from_pretrained(self.clip_model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(self.clip_model_name)

        self.clip_model.to(self.device)
        self.clip_model.eval()

        # 更新aligner的输入维度
        actual_hidden_dim = self.clip_model.config.vision_config.hidden_size
        if actual_hidden_dim != self.aligner_config.input_dim:
            print(f"Updating aligner input_dim: {self.aligner_config.input_dim} -> {actual_hidden_dim}")
            self.aligner_config.input_dim = actual_hidden_dim
            self.aligner = CLIPAligner(self.aligner_config)

        self.aligner.to(self.device)
        self.compressor.to(self.device)

        self._initialized = True

    def extract_features(self, images: list) -> torch.Tensor:
        """
        从图像列表提取CLIP特征

        Args:
            images: PIL.Image列表
        Returns:
            [num_images, seq_len, hidden_dim] vision tokens
        """
        self._init_clip()

        # 预处理图像
        inputs = self.clip_processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 提取CLIP特征
        with torch.no_grad():
            outputs = self.clip_model.vision_model(**inputs)
            vision_tokens = outputs.last_hidden_state  # [batch, seq, hidden]

        return vision_tokens

    def forward(self, images: list) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        完整的处理流程

        Args:
            images: PIL.Image列表
        Returns:
            (aligned_tokens, key_embedding)
            - aligned_tokens: [num_images, seq_len', output_dim]
            - key_embedding: [num_images, output_dim] 用于检索的向量
        """
        self._init_clip()

        # 1. 提取CLIP特征
        vision_tokens = self.extract_features(images)

        # 2. Aligner投影
        aligned_tokens = self.aligner(vision_tokens)

        # 3. 压缩（可选）
        compressed_tokens = self.compressor(aligned_tokens)

        # 4. 生成key embedding（用于检索）
        # 使用mean pooling作为整体表示
        key_embedding = aligned_tokens.mean(dim=1)

        return compressed_tokens, key_embedding

    def process_single(self, image) -> Tuple[np.ndarray, np.ndarray]:
        """
        处理单张图像，返回numpy数组

        Args:
            image: PIL.Image
        Returns:
            (tokens, key_embedding) numpy arrays
        """
        tokens, key_emb = self.forward([image])
        return tokens.cpu().numpy().squeeze(0), key_emb.cpu().numpy().squeeze(0)


def create_deepseek_style_processor(
    output_dim: int = 1024,
    compress_mode: str = "none",
    target_tokens: int = 64,
    device: str = "cuda"
) -> CLIPVisionProcessor:
    """
    创建DeepSeek风格的CLIP处理器

    Args:
        output_dim: 输出维度（通常匹配LLM的hidden_size）
        compress_mode: 压缩模式 (none, pooling, resampler, conv)
        target_tokens: 目标token数（用于resampler）
        device: 设备

    Returns:
        CLIPVisionProcessor实例
    """
    aligner_config = AlignerConfig(
        input_dim=768,  # CLIP ViT-B/32 default
        output_dim=output_dim,
        hidden_dim=output_dim,
        activation="gelu",
        use_layer_norm=True,
        add_positional_encoding=True
    )

    compressor_config = CompressorConfig(
        compress_mode=compress_mode,
        target_tokens=target_tokens,
        pooling_mode="mean"
    )

    return CLIPVisionProcessor(
        aligner_config=aligner_config,
        compressor_config=compressor_config,
        device=device
    )


__all__ = [
    "AlignerConfig",
    "CompressorConfig",
    "CLIPAligner",
    "TokenCompressor",
    "CLIPVisionProcessor",
    "create_deepseek_style_processor"
]
