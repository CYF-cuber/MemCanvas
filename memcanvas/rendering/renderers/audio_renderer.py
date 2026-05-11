"""
AudioRenderer - 音频渲染器

将音频数据转换为频谱图（spectrogram）图像。
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union
from PIL import Image
from pathlib import Path
import numpy as np

# 可选依赖
try:
    import librosa
    import librosa.display
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


@dataclass
class AudioStyle:
    """音频渲染样式配置"""
    spectrogram_type: str = "mel"  # mel, stft, cqt
    n_mels: int = 128  # mel 频带数
    n_fft: int = 2048  # FFT 窗口大小
    hop_length: int = 512  # 帧移
    colormap: str = "magma"  # matplotlib colormap
    background_color: Tuple[int, int, int, int] = (255, 255, 255, 255)
    show_colorbar: bool = False
    show_axes: bool = True
    title: Optional[str] = None


class AudioRenderer:
    """
    音频渲染器

    将音频文件或数据转换为频谱图图像。
    """

    def __init__(self, style: Optional[AudioStyle] = None):
        self.style = style or AudioStyle()

        if not LIBROSA_AVAILABLE:
            print("警告: librosa 未安装，音频渲染功能将受限")
            print("请运行: pip install librosa")

        if not MATPLOTLIB_AVAILABLE:
            print("警告: matplotlib 未安装，频谱图渲染功能将受限")
            print("请运行: pip install matplotlib")

    def render(
        self,
        source: Union[str, Path, np.ndarray],
        target_width: int = 800,
        target_height: int = 400,
        sr: Optional[int] = None
    ) -> Image.Image:
        """
        渲染音频为频谱图

        Args:
            source: 音频文件路径或 numpy 数组
            target_width: 目标宽度
            target_height: 目标高度
            sr: 采样率（仅当 source 为 numpy 数组时需要）

        Returns:
            频谱图 RGBA 图像
        """
        if not LIBROSA_AVAILABLE or not MATPLOTLIB_AVAILABLE:
            return self._render_placeholder(target_width, target_height, "需要安装 librosa 和 matplotlib")

        # 加载音频
        if isinstance(source, (str, Path)):
            y, sr = librosa.load(str(source), sr=sr)
        else:
            y = source
            if sr is None:
                sr = 22050  # 默认采样率

        # 计算频谱图
        if self.style.spectrogram_type == "mel":
            S = librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_mels=self.style.n_mels,
                n_fft=self.style.n_fft,
                hop_length=self.style.hop_length
            )
            S_db = librosa.power_to_db(S, ref=np.max)
        elif self.style.spectrogram_type == "stft":
            S = np.abs(librosa.stft(y, n_fft=self.style.n_fft, hop_length=self.style.hop_length))
            S_db = librosa.amplitude_to_db(S, ref=np.max)
        elif self.style.spectrogram_type == "cqt":
            S = np.abs(librosa.cqt(y, sr=sr, hop_length=self.style.hop_length))
            S_db = librosa.amplitude_to_db(S, ref=np.max)
        else:
            raise ValueError(f"未知的频谱图类型: {self.style.spectrogram_type}")

        # 使用 matplotlib 渲染
        return self._render_with_matplotlib(S_db, sr, target_width, target_height)

    def _render_with_matplotlib(
        self,
        S_db: np.ndarray,
        sr: int,
        width: int,
        height: int
    ) -> Image.Image:
        """使用 matplotlib 渲染频谱图"""
        # 计算 figure 尺寸（英寸）
        dpi = 100
        fig_width = width / dpi
        fig_height = height / dpi

        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)

        # 绘制频谱图
        img = librosa.display.specshow(
            S_db,
            sr=sr,
            hop_length=self.style.hop_length,
            x_axis='time' if self.style.show_axes else None,
            y_axis='mel' if self.style.show_axes else None,
            ax=ax,
            cmap=self.style.colormap
        )

        if self.style.show_colorbar:
            fig.colorbar(img, ax=ax, format='%+2.0f dB')

        if self.style.title:
            ax.set_title(self.style.title)

        if not self.style.show_axes:
            ax.axis('off')

        plt.tight_layout()

        # 转换为 PIL Image (兼容新版 matplotlib)
        fig.canvas.draw()

        # 使用 buffer_rgba() 方法（新版 matplotlib）
        try:
            buf = fig.canvas.buffer_rgba()
            data = np.asarray(buf)
            image = Image.fromarray(data, mode='RGBA')
        except AttributeError:
            # 回退到旧版方法
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            image = Image.fromarray(data, mode='RGB').convert('RGBA')

        plt.close(fig)

        # 调整到目标尺寸
        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.LANCZOS)

        return image

    def _render_placeholder(
        self,
        width: int,
        height: int,
        message: str
    ) -> Image.Image:
        """渲染占位图像"""
        from PIL import ImageDraw, ImageFont

        image = Image.new('RGBA', (width, height), self.style.background_color)
        draw = ImageDraw.Draw(image)

        # 绘制边框
        draw.rectangle([0, 0, width - 1, height - 1], outline=(200, 200, 200, 255))

        # 绘制文字
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        text = f"[音频频谱图]\n{message}"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        x = (width - text_width) // 2
        y = (height - text_height) // 2

        draw.text((x, y), text, fill=(100, 100, 100, 255), font=font)

        return image

    def render_waveform(
        self,
        source: Union[str, Path, np.ndarray],
        target_width: int = 800,
        target_height: int = 200,
        sr: Optional[int] = None,
        color: str = "#1f77b4"
    ) -> Image.Image:
        """
        渲染波形图

        Args:
            source: 音频源
            target_width: 目标宽度
            target_height: 目标高度
            sr: 采样率
            color: 波形颜色

        Returns:
            波形图 RGBA 图像
        """
        if not LIBROSA_AVAILABLE or not MATPLOTLIB_AVAILABLE:
            return self._render_placeholder(target_width, target_height, "需要安装 librosa 和 matplotlib")

        # 加载音频
        if isinstance(source, (str, Path)):
            y, sr = librosa.load(str(source), sr=sr)
        else:
            y = source
            if sr is None:
                sr = 22050

        # 计算 figure 尺寸
        dpi = 100
        fig_width = target_width / dpi
        fig_height = target_height / dpi

        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)

        # 绘制波形
        librosa.display.waveshow(y, sr=sr, ax=ax, color=color)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')

        plt.tight_layout()

        # 转换为 PIL Image (兼容新版 matplotlib)
        fig.canvas.draw()

        try:
            buf = fig.canvas.buffer_rgba()
            data = np.asarray(buf)
            image = Image.fromarray(data, mode='RGBA')
        except AttributeError:
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            image = Image.fromarray(data, mode='RGB').convert('RGBA')

        plt.close(fig)

        if image.size != (target_width, target_height):
            image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)

        return image

    def get_audio_info(self, source: Union[str, Path]) -> dict:
        """
        获取音频信息

        Args:
            source: 音频文件路径

        Returns:
            包含时长、采样率等信息的字典
        """
        if not LIBROSA_AVAILABLE:
            return {"error": "librosa 未安装"}

        y, sr = librosa.load(str(source), sr=None)
        duration = librosa.get_duration(y=y, sr=sr)

        return {
            "duration_seconds": duration,
            "sample_rate": sr,
            "num_samples": len(y),
            "channels": 1 if len(y.shape) == 1 else y.shape[0]
        }
