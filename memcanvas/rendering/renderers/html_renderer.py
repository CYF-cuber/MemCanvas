"""
HtmlRenderer - HTML/Markdown 渲染器

使用 Playwright 将 Markdown/HTML 渲染为高质量 PIL Image。
基于 MemOCR 的 Playwright 方案，调整为更紧凑排版。

Playwright 为可选依赖，导入失败时给出明确提示。
"""

from dataclasses import dataclass, field
from typing import List, Optional
from PIL import Image
import io

try:
    from playwright.sync_api import sync_playwright, Browser, Playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False


@dataclass
class HtmlStyle:
    """HTML 渲染样式配置"""
    viewport_width: int = 600
    font_size_px: int = 14
    font_family: str = (
        "'Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'DejaVu Sans', "
        "Arial, Helvetica, sans-serif"
    )
    margin: str = "0.5em"
    padding: str = "0.8em"
    background: str = "white"


# Compact HTML/CSS template
_HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
body {{
    background-color: {background};
    margin: {margin};
    padding: {padding};
    font-family: {font_family};
    font-size: {font_size_px}px;
    line-height: 1.4;
    color: #111;
}}
.content {{
    padding: 0;
    display: inline-block;
    white-space: pre-wrap;
    max-width: 100%;
    word-wrap: break-word;
}}
h1, h2, h3, h4, h5, h6 {{
    margin-top: 0.4em;
    margin-bottom: 0.3em;
    padding: 0;
}}
h1 {{ font-size: 1.4em; }}
h2 {{ font-size: 1.2em; }}
h3 {{ font-size: 1.1em; }}
p {{
    margin-top: 0.3em;
    margin-bottom: 0.3em;
    padding: 0;
}}
ul, ol {{
    margin-top: 0.2em;
    margin-bottom: 0.2em;
    margin-left: 18px;
    padding: 0;
}}
li {{
    margin-bottom: 0.1em;
}}
code {{
    background-color: #f4f4f4;
    padding: 1px 3px;
    border-radius: 3px;
    font-family: 'Courier New', monospace;
    font-size: 0.9em;
}}
pre {{
    background-color: #f4f4f4;
    padding: 8px;
    border-radius: 4px;
    overflow-x: auto;
    margin: 0.3em 0;
}}
blockquote {{
    border-left: 3px solid #ccc;
    margin: 0.3em 0;
    padding: 0.2em 0.6em;
    color: #555;
}}
strong {{ font-weight: bold; }}
em {{ font-style: italic; }}
hr {{
    border: none;
    border-top: 1px solid #ddd;
    margin: 0.4em 0;
}}
</style>
</head>
<body>
<div class="content">{html_content}</div>
</body>
</html>"""


class HtmlRenderer:
    """
    HTML/Markdown 渲染器

    使用 Playwright 将 Markdown 或 HTML 渲染为 PIL Image。
    内部维护单例 browser 实例以避免重复启动。

    Usage:
        renderer = HtmlRenderer()
        img = renderer.render("# Hello\\n- item 1\\n- **bold** item")
        renderer.close()
    """

    _instance_pw: Optional[object] = None  # Playwright context manager
    _instance_browser: Optional[object] = None  # Browser instance

    def __init__(self, style: Optional[HtmlStyle] = None):
        if not PLAYWRIGHT_AVAILABLE:
            raise ImportError(
                "Playwright is required for HtmlRenderer. "
                "Install it with: pip install playwright && playwright install chromium"
            )
        self.style = style or HtmlStyle()

    def _ensure_browser(self):
        """确保 browser 已启动（单例）"""
        if HtmlRenderer._instance_browser is None:
            HtmlRenderer._instance_pw = sync_playwright().start()
            HtmlRenderer._instance_browser = HtmlRenderer._instance_pw.chromium.launch(
                headless=True,
                args=["--disable-dev-shm-usage", "--disable-gpu", "--no-sandbox"],
            )

    def _build_html(self, content: str, content_type: str = "markdown") -> str:
        """将 content 转换为完整 HTML 页面"""
        if content_type == "markdown":
            import markdown
            html_content = markdown.markdown(
                content.strip().strip("`"),
                extensions=["extra"],
            )
        elif content_type == "html":
            html_content = content
        else:
            # plain text: escape and wrap in <pre>
            import html as html_mod
            html_content = f"<pre>{html_mod.escape(content)}</pre>"

        return _HTML_TEMPLATE.format(
            background=self.style.background,
            margin=self.style.margin,
            padding=self.style.padding,
            font_family=self.style.font_family,
            font_size_px=self.style.font_size_px,
            html_content=html_content,
        )

    def render(self, content: str, content_type: str = "markdown") -> Image.Image:
        """
        渲染内容为 PIL Image

        Args:
            content: Markdown、HTML 或纯文本
            content_type: "markdown", "html", 或 "text"

        Returns:
            PIL Image (RGB)
        """
        self._ensure_browser()
        full_html = self._build_html(content, content_type)

        page = HtmlRenderer._instance_browser.new_page(
            viewport={"width": self.style.viewport_width, "height": 800}
        )
        try:
            page.set_content(full_html)
            page.wait_for_load_state("domcontentloaded", timeout=5000)

            element = page.locator(".content")
            box = element.bounding_box()

            if box and box["width"] >= 10 and box["height"] >= 10:
                screenshot_bytes = page.screenshot(
                    clip={
                        "x": 0,
                        "y": 0,
                        "width": self.style.viewport_width,
                        "height": box["y"] + box["height"] + 10,
                    },
                    omit_background=False,
                    type="png",
                )
            else:
                screenshot_bytes = page.screenshot(
                    full_page=True, omit_background=False, type="png"
                )
        finally:
            page.close()

        img = Image.open(io.BytesIO(screenshot_bytes)).convert("RGB")
        return img

    def render_to_patches(
        self,
        content: str,
        content_type: str = "markdown",
        patch_height: int = 580,
    ) -> List[Image.Image]:
        """
        渲染内容，超高时自动分页

        Args:
            content: 要渲染的内容
            content_type: "markdown", "html", 或 "text"
            patch_height: 每个 patch 的最大高度

        Returns:
            PIL Image 列表（每个不超过 patch_height 高度）
        """
        full_img = self.render(content, content_type)
        w, h = full_img.size

        if h <= patch_height:
            return [full_img]

        patches = []
        y = 0
        while y < h:
            bottom = min(y + patch_height, h)
            patch = full_img.crop((0, y, w, bottom))
            patches.append(patch)
            y = bottom

        return patches

    def close(self):
        """关闭 browser（释放资源）"""
        if HtmlRenderer._instance_browser is not None:
            try:
                HtmlRenderer._instance_browser.close()
            except Exception:
                pass
            HtmlRenderer._instance_browser = None
        if HtmlRenderer._instance_pw is not None:
            try:
                HtmlRenderer._instance_pw.stop()
            except Exception:
                pass
            HtmlRenderer._instance_pw = None
