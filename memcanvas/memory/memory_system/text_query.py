"""
TextQueryEncoder - 文本查询编码器

将文本查询转换为可与记忆库key embedding比较的向量。

支持两种方式：
1. CLIP Text Encoder（推荐，直接使用CLIP文本编码器）
2. Canvas Render（将文本渲染到画布，再用CLIP Vision编码）
"""

from dataclasses import dataclass
from typing import Optional, Literal, Tuple
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass
class TextQueryConfig:
    """文本查询配置"""
    # 编码方式: clip_text, canvas_render
    encode_mode: str = "clip_text"
    # CLIP模型
    model_name: str = "openai/clip-vit-base-patch32"
    # 设备
    device: str = "cuda"
    # Canvas渲染尺寸
    canvas_size: Tuple[int, int] = (512, 512)
    # 是否使用Aligner（canvas_render模式）
    use_aligner: bool = True
    # Aligner输出维度
    aligner_output_dim: int = 1024


class TextQueryEncoder:
    """
    文本查询编码器

    将文本查询转换为向量，用于在记忆库中检索。

    使用示例：
    ```python
    encoder = TextQueryEncoder()

    # 方式1: CLIP文本编码（推荐）
    query_vector = encoder.encode("会议记录")

    # 方式2: Canvas渲染后编码
    query_vector = encoder.encode_via_canvas("会议记录")

    # 在记忆库中检索
    results = manager.retrieve(query_vector=query_vector, top_k=5)
    ```
    """

    def __init__(self, config: Optional[TextQueryConfig] = None):
        self.config = config or TextQueryConfig()

        self._clip_model = None
        self._clip_processor = None
        self._aligner = None
        self._initialized = False

    def _init_model(self):
        """延迟初始化模型"""
        if self._initialized:
            return

        try:
            from transformers import CLIPModel, CLIPProcessor

            self._clip_model = CLIPModel.from_pretrained(self.config.model_name)
            self._clip_processor = CLIPProcessor.from_pretrained(self.config.model_name)
            self._clip_model.to(self.config.device)
            self._clip_model.eval()

            # 如果需要Aligner
            if self.config.use_aligner and self.config.encode_mode == "canvas_render":
                self._init_aligner()

            self._initialized = True

        except ImportError:
            raise ImportError("请安装transformers: pip install transformers")

    def _init_aligner(self):
        """初始化Aligner"""
        from ...encoders.slicer.clip_aligner import CLIPAligner, AlignerConfig

        input_dim = self._clip_model.config.vision_config.hidden_size

        aligner_config = AlignerConfig(
            input_dim=input_dim,
            output_dim=self.config.aligner_output_dim,
            activation="gelu",
            use_layer_norm=True
        )

        self._aligner = CLIPAligner(aligner_config)
        self._aligner.to(self.config.device)
        self._aligner.eval()

    def encode(self, text: str) -> np.ndarray:
        """
        编码文本查询（使用配置的默认方式）

        Args:
            text: 查询文本

        Returns:
            查询向量 [dim]
        """
        if self.config.encode_mode == "clip_text":
            return self.encode_via_clip_text(text)
        else:
            return self.encode_via_canvas(text)

    def encode_via_clip_text(self, text: str) -> np.ndarray:
        """
        使用CLIP Text Encoder编码文本

        这是推荐的方式，因为CLIP的文本和图像编码器输出在同一语义空间。

        Args:
            text: 查询文本

        Returns:
            查询向量 [dim]
        """
        import torch

        self._init_model()

        # 处理文本
        inputs = self._clip_processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        inputs = {k: v.to(self.config.device) for k, v in inputs.items() if k != "pixel_values"}

        with torch.no_grad():
            # 获取文本特征
            text_features = self._clip_model.get_text_features(**inputs)

            # 归一化
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features.cpu().numpy().squeeze(0)

    def encode_via_canvas(self, text: str) -> np.ndarray:
        """
        将文本渲染到Canvas，然后用CLIP Vision编码

        流程：文本 → Canvas渲染 → CLIP Vision → (可选Aligner) → 向量

        Args:
            text: 查询文本

        Returns:
            查询向量 [dim]
        """
        import torch

        self._init_model()

        # 1. 渲染文本到Canvas
        canvas_image = self._render_text_to_canvas(text)

        # 2. CLIP Vision编码
        inputs = self._clip_processor(images=canvas_image, return_tensors="pt")
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}

        with torch.no_grad():
            # 获取vision features
            outputs = self._clip_model.vision_model(**inputs)
            vision_features = outputs.last_hidden_state  # [1, seq, hidden]

            # 3. 可选：通过Aligner
            if self._aligner is not None:
                vision_features = self._aligner(vision_features)

            # 4. Mean pooling得到单一向量
            query_vector = vision_features.mean(dim=1)  # [1, dim]

            # 归一化
            query_vector = query_vector / query_vector.norm(dim=-1, keepdim=True)

        return query_vector.cpu().numpy().squeeze(0)

    def _render_text_to_canvas(self, text: str) -> Image.Image:
        """将文本渲染到Canvas"""
        width, height = self.config.canvas_size

        # 创建白色画布
        canvas = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(canvas)

        # 尝试使用字体
        try:
            # 尝试加载中文字体
            font_paths = [
                "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/System/Library/Fonts/PingFang.ttc",
            ]
            font = None
            for fp in font_paths:
                if Path(fp).exists():
                    font = ImageFont.truetype(fp, 24)
                    break
            if font is None:
                font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()

        # 文本换行处理
        margin = 30
        max_width = width - 2 * margin

        lines = []
        current_line = ""
        for char in text:
            test_line = current_line + char
            bbox = draw.textbbox((0, 0), test_line, font=font)
            if bbox[2] - bbox[0] <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = char
        if current_line:
            lines.append(current_line)

        # 绘制文本
        y = margin
        for line in lines:
            draw.text((margin, y), line, fill='black', font=font)
            bbox = draw.textbbox((0, 0), line, font=font)
            y += (bbox[3] - bbox[1]) + 10

            if y > height - margin:
                break

        return canvas

    def encode_batch(self, texts: list) -> np.ndarray:
        """
        批量编码文本

        Args:
            texts: 文本列表

        Returns:
            向量数组 [batch, dim]
        """
        import torch

        self._init_model()

        if self.config.encode_mode == "clip_text":
            inputs = self._clip_processor(
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            inputs = {k: v.to(self.config.device) for k, v in inputs.items() if k != "pixel_values"}

            with torch.no_grad():
                text_features = self._clip_model.get_text_features(**inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            return text_features.cpu().numpy()
        else:
            # Canvas模式需要逐个处理
            vectors = [self.encode_via_canvas(t) for t in texts]
            return np.stack(vectors)

    def get_canvas_preview(self, text: str) -> Image.Image:
        """获取Canvas渲染预览"""
        return self._render_text_to_canvas(text)


def create_text_encoder(
    mode: Literal["clip_text", "canvas_render"] = "clip_text",
    device: str = "cuda"
) -> TextQueryEncoder:
    """
    快速创建文本编码器

    Args:
        mode: 编码模式
            - "clip_text": 使用CLIP文本编码器（推荐）
            - "canvas_render": 渲染到画布后用视觉编码器
        device: 设备

    Returns:
        TextQueryEncoder实例
    """
    config = TextQueryConfig(
        encode_mode=mode,
        device=device
    )
    return TextQueryEncoder(config)
