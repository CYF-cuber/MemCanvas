"""
TextQueryEncoder - text

textkey embeddingtextvector。

text：
1. CLIP Text Encoder（text，textCLIPtext）
2. Canvas Render（text，textCLIP Visiontext）
"""

from dataclasses import dataclass
from typing import Optional, Literal, Tuple
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass
class TextQueryConfig:
    """text"""
    # text: clip_text, canvas_render
    encode_mode: str = "clip_text"
    # CLIPtext
    model_name: str = "openai/clip-vit-base-patch32"
    # text
    device: str = "cuda"
    # Canvastext
    canvas_size: Tuple[int, int] = (512, 512)
    # textAligner（canvas_rendertext）
    use_aligner: bool = True
    # Alignertext
    aligner_output_dim: int = 1024


class TextQueryEncoder:
    """
    text

    textvector，text。

    text：
    ```python
    encoder = TextQueryEncoder()

    # text1: CLIPtext（text）
    query_vector = encoder.encode("text")

    # text2: Canvastext
    query_vector = encoder.encode_via_canvas("text")

    # text
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
        """text"""
        if self._initialized:
            return

        try:
            from transformers import CLIPModel, CLIPProcessor

            self._clip_model = CLIPModel.from_pretrained(self.config.model_name)
            self._clip_processor = CLIPProcessor.from_pretrained(self.config.model_name)
            self._clip_model.to(self.config.device)
            self._clip_model.eval()

            # textAligner
            if self.config.use_aligner and self.config.encode_mode == "canvas_render":
                self._init_aligner()

            self._initialized = True

        except ImportError:
            raise ImportError("texttransformers: pip install transformers")

    def _init_aligner(self):
        """textAligner"""
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
        text（text）

        Args:
            text: text

        Returns:
            textvector [dim]
        """
        if self.config.encode_mode == "clip_text":
            return self.encode_via_clip_text(text)
        else:
            return self.encode_via_canvas(text)

    def encode_via_clip_text(self, text: str) -> np.ndarray:
        """
        textCLIP Text Encodertext

        text，textCLIPtext。

        Args:
            text: text

        Returns:
            textvector [dim]
        """
        import torch

        self._init_model()

        # text
        inputs = self._clip_processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        inputs = {k: v.to(self.config.device) for k, v in inputs.items() if k != "pixel_values"}

        with torch.no_grad():
            # text
            text_features = self._clip_model.get_text_features(**inputs)

            # text
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features.cpu().numpy().squeeze(0)

    def encode_via_canvas(self, text: str) -> np.ndarray:
        """
        textCanvas，textCLIP Visiontext

        text：text → Canvastext → CLIP Vision → (textAligner) → vector

        Args:
            text: text

        Returns:
            textvector [dim]
        """
        import torch

        self._init_model()

        # 1. textCanvas
        canvas_image = self._render_text_to_canvas(text)

        # 2. CLIP Visiontext
        inputs = self._clip_processor(images=canvas_image, return_tensors="pt")
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}

        with torch.no_grad():
            # textvision features
            outputs = self._clip_model.vision_model(**inputs)
            vision_features = outputs.last_hidden_state  # [1, seq, hidden]

            # 3. text：textAligner
            if self._aligner is not None:
                vision_features = self._aligner(vision_features)

            # 4. Mean poolingtextvector
            query_vector = vision_features.mean(dim=1)  # [1, dim]

            # text
            query_vector = query_vector / query_vector.norm(dim=-1, keepdim=True)

        return query_vector.cpu().numpy().squeeze(0)

    def _render_text_to_canvas(self, text: str) -> Image.Image:
        """textCanvas"""
        width, height = self.config.canvas_size

        # text
        canvas = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(canvas)

        # text
        try:
            # text
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

        # text
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

        # text
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
        text

        Args:
            texts: text

        Returns:
            vectortext [batch, dim]
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
            # Canvastext
            vectors = [self.encode_via_canvas(t) for t in texts]
            return np.stack(vectors)

    def get_canvas_preview(self, text: str) -> Image.Image:
        """textCanvastext"""
        return self._render_text_to_canvas(text)


def create_text_encoder(
    mode: Literal["clip_text", "canvas_render"] = "clip_text",
    device: str = "cuda"
) -> TextQueryEncoder:
    """
    text

    Args:
        mode: text
            - "clip_text": textCLIPtext（text）
            - "canvas_render": text
        device: text

    Returns:
        TextQueryEncodertext
    """
    config = TextQueryConfig(
        encode_mode=mode,
        device=device
    )
    return TextQueryEncoder(config)
