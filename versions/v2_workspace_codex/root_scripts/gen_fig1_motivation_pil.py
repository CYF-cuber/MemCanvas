"""
Generate Figure 1 (Motivation) for MemCanvas paper using PIL.

Layout (top to bottom):
  Row 1: Historical conversation — user uploads Shakespeare portrait, asks about the person
  Row 2: Memory storage — left: text-only memory; right: MemCanvas (image + text)
  Row 3: Later query — text memory fails (red X), MemCanvas succeeds (green check)
"""

from PIL import Image, ImageDraw, ImageFont
import math

# ── Fonts ──
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_SERIF_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"
FONT_CJK_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

def font(size): return ImageFont.truetype(FONT_PATH, size)
def font_bold(size): return ImageFont.truetype(FONT_BOLD_PATH, size)
def font_serif(size): return ImageFont.truetype(FONT_SERIF_PATH, size)

# ── Colors ──
WHITE = (255, 255, 255)
BG = (250, 250, 252)
LIGHT_BLUE = (230, 240, 255)
LIGHT_GRAY = (240, 242, 245)
MID_GRAY = (180, 185, 195)
DARK_GRAY = (80, 85, 95)
TEXT_COLOR = (40, 42, 48)
BLUE = (65, 105, 180)
LIGHT_BLUE_BG = (235, 244, 255)
GREEN = (34, 139, 34)
GREEN_BG = (230, 255, 230)
RED = (200, 50, 50)
RED_BG = (255, 230, 230)
ORANGE = (200, 130, 50)
CANVAS_BORDER = (100, 120, 180)
USER_COLOR = (70, 130, 200)
AGENT_COLOR = (100, 100, 115)
ARROW_COLOR = (130, 140, 160)
SECTION_BG = (245, 247, 252)
MEMCANVAS_ACCENT = (50, 100, 200)
TEXTMEM_ACCENT = (160, 90, 50)

W, H = 1600, 1100


def draw_rounded_rect(draw, bbox, radius, fill=None, outline=None, width=1):
    x0, y0, x1, y1 = bbox
    r = radius
    if fill:
        draw.rectangle([x0 + r, y0, x1 - r, y1], fill=fill)
        draw.rectangle([x0, y0 + r, x1, y1 - r], fill=fill)
        draw.pieslice([x0, y0, x0 + 2*r, y0 + 2*r], 180, 270, fill=fill)
        draw.pieslice([x1 - 2*r, y0, x1, y0 + 2*r], 270, 360, fill=fill)
        draw.pieslice([x0, y1 - 2*r, x0 + 2*r, y1], 90, 180, fill=fill)
        draw.pieslice([x1 - 2*r, y1 - 2*r, x1, y1], 0, 90, fill=fill)
    if outline:
        draw.arc([x0, y0, x0 + 2*r, y0 + 2*r], 180, 270, fill=outline, width=width)
        draw.arc([x1 - 2*r, y0, x1, y0 + 2*r], 270, 360, fill=outline, width=width)
        draw.arc([x0, y1 - 2*r, x0 + 2*r, y1], 90, 180, fill=outline, width=width)
        draw.arc([x1 - 2*r, y1 - 2*r, x1, y1], 0, 90, fill=outline, width=width)
        draw.line([x0 + r, y0, x1 - r, y0], fill=outline, width=width)
        draw.line([x0 + r, y1, x1 - r, y1], fill=outline, width=width)
        draw.line([x0, y0 + r, x0, y1 - r], fill=outline, width=width)
        draw.line([x1, y0 + r, x1, y1 - r], fill=outline, width=width)


def draw_arrow(draw, x0, y0, x1, y1, color=ARROW_COLOR, width=2, head_size=10):
    draw.line([(x0, y0), (x1, y1)], fill=color, width=width)
    angle = math.atan2(y1 - y0, x1 - x0)
    lx = x1 - head_size * math.cos(angle - 0.4)
    ly = y1 - head_size * math.sin(angle - 0.4)
    rx = x1 - head_size * math.cos(angle + 0.4)
    ry = y1 - head_size * math.sin(angle + 0.4)
    draw.polygon([(x1, y1), (lx, ly), (rx, ry)], fill=color)


def draw_chat_bubble(draw, x, y, w, h, text, text_font, color, text_color, tail_side="left"):
    r = 12
    draw_rounded_rect(draw, (x, y, x+w, y+h), r, fill=color, outline=None)
    # tail
    if tail_side == "left":
        draw.polygon([(x+8, y+h), (x-8, y+h+8), (x+18, y+h)], fill=color)
    else:
        draw.polygon([(x+w-8, y+h), (x+w+8, y+h+8), (x+w-18, y+h)], fill=color)
    # text
    lines = text.split('\n')
    ty = y + 8
    for line in lines:
        draw.text((x + 12, ty), line, font=text_font, fill=text_color)
        ty += text_font.size + 6


def draw_portrait_placeholder(img, x, y, w, h):
    """Draw a stylized Renaissance portrait placeholder."""
    draw = ImageDraw.Draw(img)
    # Dark background like oil painting
    draw_rounded_rect(draw, (x, y, x+w, y+h), 6, fill=(45, 35, 30), outline=(80, 65, 50), width=2)

    cx, cy = x + w//2, y + h//2 - 10

    # Face (oval)
    face_w, face_h = w//3, int(h*0.32)
    draw.ellipse([cx-face_w, cy-face_h, cx+face_w, cy+face_h], fill=(210, 180, 155))

    # Hair (dark, receding, Renaissance style)
    draw.ellipse([cx-face_w-6, cy-face_h-8, cx+face_w+6, cy-face_h//2+5], fill=(60, 40, 25))
    draw.ellipse([cx-face_w-10, cy-face_h//2-5, cx-face_w+8, cy+face_h//2], fill=(60, 40, 25))
    draw.ellipse([cx+face_w-8, cy-face_h//2-5, cx+face_w+10, cy+face_h//2], fill=(60, 40, 25))

    # Eyes
    ey = cy - face_h//5
    for ex_off in [-face_w//2, face_w//2]:
        draw.ellipse([cx+ex_off-5, ey-3, cx+ex_off+5, ey+3], fill=(40, 30, 20))

    # Nose
    draw.line([(cx, cy-3), (cx, cy+face_h//5)], fill=(180, 150, 130), width=2)

    # Mouth
    my = cy + face_h//3
    draw.arc([cx-10, my-3, cx+10, my+6], 0, 180, fill=(170, 120, 100), width=2)

    # Beard (small goatee)
    draw.ellipse([cx-12, my+3, cx+12, my+18], fill=(60, 40, 25))

    # White ruff collar — THE KEY DETAIL
    collar_y = cy + face_h + 2
    collar_w = face_w + 25
    # Draw layered white ruff
    for i in range(5):
        offset = i * 6
        draw.ellipse([cx-collar_w+offset, collar_y-3+i*4, cx+collar_w-offset, collar_y+18+i*3],
                     fill=None, outline=(245, 242, 235), width=3)
    # Fill collar area
    draw.ellipse([cx-collar_w, collar_y-2, cx+collar_w, collar_y+25],
                 fill=None, outline=(240, 238, 230), width=4)
    for i in range(8):
        angle = math.pi * i / 7
        rx, ry = collar_w * math.cos(angle), 14 * math.sin(angle)
        draw.ellipse([cx-rx-4, collar_y+ry-4+8, cx-rx+4, collar_y+ry+4+8],
                     fill=(250, 248, 242))

    # Dark doublet (body)
    body_top = collar_y + 22
    if body_top < y + h - 5:
        draw.rectangle([cx-face_w-15, body_top, cx+face_w+15, y+h-5], fill=(30, 25, 22))

    # Gold earring (small detail)
    draw.ellipse([cx+face_w+2, ey+10, cx+face_w+8, ey+20], fill=(200, 170, 80))

    # "Portrait" label
    small_f = font(11)
    draw.text((x+5, y+h-18), "Portrait", font=small_f, fill=(150, 140, 120))


def draw_user_icon(draw, cx, cy, r, color=USER_COLOR):
    # Head
    draw.ellipse([cx-r//3, cy-r, cx+r//3, cy-r//3], fill=color)
    # Body
    draw.ellipse([cx-r, cy-r//6, cx+r, cy+r], fill=color)


def draw_doc_icon(draw, x, y, w, h, color=MID_GRAY):
    fold = 12
    points = [(x, y), (x+w-fold, y), (x+w, y+fold), (x+w, y+h), (x, y+h)]
    draw.polygon(points, fill=WHITE, outline=color)
    draw.polygon([(x+w-fold, y), (x+w-fold, y+fold), (x+w, y+fold)], fill=(220, 225, 235), outline=color)
    # Lines
    for i in range(4):
        ly = y + fold + 12 + i * 14
        if ly + 5 < y + h - 8:
            draw.line([(x+10, ly), (x+w-12, ly)], fill=(180, 185, 195), width=2)


def draw_section_label(draw, text, x, y, color=DARK_GRAY):
    f = font_bold(17)
    draw.text((x, y), text, font=f, fill=color)


def create_figure():
    img = Image.new("RGB", (W, H), WHITE)
    draw = ImageDraw.Draw(img)

    # ══════════════════════════════════════════════════════════
    # ROW 1: Historical Conversation
    # ══════════════════════════════════════════════════════════
    row1_top = 15
    row1_h = 280
    row1_box = (30, row1_top, W-30, row1_top + row1_h)
    draw_rounded_rect(draw, row1_box, 16, fill=(248, 250, 255), outline=(200, 210, 230), width=2)

    # Section label
    label_f = font_bold(18)
    draw.text((50, row1_top + 12), "Step 1", font=label_f, fill=BLUE)
    draw.text((110, row1_top + 12), "  Historical Conversation", font=font(18), fill=DARK_GRAY)

    # User icon
    user_cx, user_cy = 100, row1_top + 90
    draw_user_icon(draw, user_cx, user_cy, 20, USER_COLOR)
    draw.text((user_cx - 15, user_cy + 25), "User", font=font(13), fill=USER_COLOR)

    # Portrait (user uploads)
    portrait_x, portrait_y = 145, row1_top + 52
    portrait_w, portrait_h = 130, 170
    draw_portrait_placeholder(img, portrait_x, portrait_y, portrait_w, portrait_h)

    # Upload indicator
    draw.text((portrait_x + 2, portrait_y + portrait_h + 4), "📎 uploaded image", font=font(12), fill=MID_GRAY)

    # User chat bubble
    bx, by = 310, row1_top + 55
    user_msg = "Who is this person?\nTell me about his life."
    draw_chat_bubble(draw, bx, by, 280, 55, user_msg, font(15), LIGHT_BLUE_BG, TEXT_COLOR, "left")

    # Agent icon
    agent_cx = 310
    agent_cy = row1_top + 155
    draw.ellipse([agent_cx-18, agent_cy-18, agent_cx+18, agent_cy+18], fill=AGENT_COLOR)
    draw.text((agent_cx-6, agent_cy-9), "AI", font=font_bold(13), fill=WHITE)
    draw.text((agent_cx - 18, agent_cy + 22), "Agent", font=font(13), fill=AGENT_COLOR)

    # Agent chat bubble
    abx, aby = 355, row1_top + 135
    agent_msg = "This is William Shakespeare (1564-1616),\nan English playwright and poet, widely\nregarded as the greatest writer in the\nEnglish language..."
    draw_chat_bubble(draw, abx, aby, 380, 95, agent_msg, font(14), (240, 242, 248), TEXT_COLOR, "left")

    # Right side: visual summary of conversation
    summary_x = 800
    draw_rounded_rect(draw, (summary_x, row1_top + 48, W-50, row1_top + row1_h - 20), 12,
                      fill=(252, 253, 255), outline=(210, 215, 230), width=1)
    draw.text((summary_x + 15, row1_top + 55), "Conversation contains:", font=font_bold(14), fill=DARK_GRAY)
    draw.text((summary_x + 15, row1_top + 80), "• Visual content (portrait image)", font=font(14), fill=(50, 120, 50))
    draw.text((summary_x + 15, row1_top + 102), "• Textual content (biography Q&A)", font=font(14), fill=DARK_GRAY)
    draw.text((summary_x + 15, row1_top + 132), "How should the agent memorize", font=font_bold(14), fill=ORANGE)
    draw.text((summary_x + 15, row1_top + 154), "this multimodal conversation?", font=font_bold(14), fill=ORANGE)

    # Small portrait in summary
    sp_x, sp_y = W - 200, row1_top + 80
    draw_portrait_placeholder(img, sp_x, sp_y, 80, 105)
    # Plus text icon
    draw.text((sp_x - 40, sp_y + 30), "+", font=font_bold(30), fill=ORANGE)
    draw_doc_icon(draw, sp_x - 75, sp_y + 15, 35, 45, MID_GRAY)

    # ══════════════════════════════════════════════════════════
    # Arrows from Row 1 to Row 2
    # ══════════════════════════════════════════════════════════
    arrow_y_start = row1_top + row1_h + 5
    arrow_y_end = row1_top + row1_h + 42

    # Left arrow (text memory)
    left_center = W // 4
    draw_arrow(draw, left_center, arrow_y_start, left_center, arrow_y_end, TEXTMEM_ACCENT, 3, 12)
    draw.text((left_center - 50, arrow_y_start + 6), "store to memory", font=font(13), fill=TEXTMEM_ACCENT)

    # Right arrow (memcanvas)
    right_center = 3 * W // 4
    draw_arrow(draw, right_center, arrow_y_start, right_center, arrow_y_end, MEMCANVAS_ACCENT, 3, 12)
    draw.text((right_center - 50, arrow_y_start + 6), "store to memory", font=font(13), fill=MEMCANVAS_ACCENT)

    # ══════════════════════════════════════════════════════════
    # ROW 2: Memory Storage (two paths)
    # ══════════════════════════════════════════════════════════
    row2_top = arrow_y_end + 5
    row2_h = 340
    mid_x = W // 2

    # Dashed vertical divider
    for dy in range(0, row2_h, 10):
        draw.line([(mid_x, row2_top + dy), (mid_x, row2_top + dy + 5)], fill=MID_GRAY, width=2)

    # ── LEFT: Text-based Memory ──
    left_box = (40, row2_top, mid_x - 25, row2_top + row2_h)
    draw_rounded_rect(draw, left_box, 14, fill=(255, 250, 245), outline=TEXTMEM_ACCENT, width=2)

    # Label
    draw.text((60, row2_top + 10), "Text-based Memory", font=font_bold(18), fill=TEXTMEM_ACCENT)
    draw.text((290, row2_top + 12), "(Baseline: Mem0 / RAG)", font=font(14), fill=(150, 130, 110))

    # Document icon
    doc_x, doc_y = 80, row2_top + 50
    draw_doc_icon(draw, doc_x, doc_y, 100, 130, TEXTMEM_ACCENT)
    # Text inside doc
    doc_text_x = doc_x + 12
    draw.text((doc_x + 8, doc_y + 18), "Conversation", font=font_bold(11), fill=TEXTMEM_ACCENT)
    draw.text((doc_x + 8, doc_y + 34), "Summary:", font=font_bold(11), fill=TEXTMEM_ACCENT)
    draw.text((doc_x + 8, doc_y + 55), "User asked", font=font(10), fill=DARK_GRAY)
    draw.text((doc_x + 8, doc_y + 68), "about a person", font=font(10), fill=DARK_GRAY)
    draw.text((doc_x + 8, doc_y + 81), "in a portrait.", font=font(10), fill=DARK_GRAY)
    draw.text((doc_x + 8, doc_y + 94), "Shakespeare,", font=font(10), fill=DARK_GRAY)
    draw.text((doc_x + 8, doc_y + 107), "1564-1616 ...", font=font(10), fill=DARK_GRAY)

    # Annotation: what's stored
    anno_x = 220
    draw.text((anno_x, row2_top + 55), "Stored:", font=font_bold(15), fill=DARK_GRAY)
    draw.text((anno_x, row2_top + 78), "✓  Textual summary of conversation", font=font(14), fill=(80, 130, 80))
    draw.text((anno_x, row2_top + 100), "✓  Key entities (Shakespeare, dates)", font=font(14), fill=(80, 130, 80))

    draw.text((anno_x, row2_top + 135), "Lost:", font=font_bold(15), fill=RED)
    draw.text((anno_x, row2_top + 158), "✗  The portrait image itself", font=font(14), fill=RED)
    draw.text((anno_x, row2_top + 180), "✗  Visual details (clothing, colors)", font=font(14), fill=RED)
    draw.text((anno_x, row2_top + 202), "✗  Spatial layout information", font=font(14), fill=RED)

    # Big X over image
    img_lost_x, img_lost_y = 100, row2_top + 220
    draw_rounded_rect(draw, (img_lost_x, img_lost_y, img_lost_x+90, img_lost_y+70), 6,
                      fill=(245, 235, 230), outline=(200, 180, 170), width=1)
    draw.text((img_lost_x+10, img_lost_y+10), "🖼️ Image", font=font(14), fill=MID_GRAY)
    draw.text((img_lost_x+15, img_lost_y+32), "  data", font=font(14), fill=MID_GRAY)
    draw.text((img_lost_x+10, img_lost_y+50), "DISCARDED", font=font_bold(10), fill=RED)
    # Red X
    draw.line([(img_lost_x, img_lost_y), (img_lost_x+90, img_lost_y+70)], fill=RED, width=3)
    draw.line([(img_lost_x+90, img_lost_y), (img_lost_x, img_lost_y+70)], fill=RED, width=3)

    # ── RIGHT: MemCanvas ──
    right_box = (mid_x + 25, row2_top, W - 40, row2_top + row2_h)
    draw_rounded_rect(draw, right_box, 14, fill=(240, 245, 255), outline=MEMCANVAS_ACCENT, width=2)

    # Label
    draw.text((mid_x + 50, row2_top + 10), "MemCanvas", font=font_bold(18), fill=MEMCANVAS_ACCENT)
    draw.text((mid_x + 200, row2_top + 12), "(Ours)", font=font_bold(14), fill=MEMCANVAS_ACCENT)

    # Draw a mini canvas with portrait + text
    canvas_x, canvas_y = mid_x + 60, row2_top + 48
    canvas_w, canvas_h = 300, 265
    draw_rounded_rect(draw, (canvas_x, canvas_y, canvas_x+canvas_w, canvas_y+canvas_h), 8,
                      fill=WHITE, outline=CANVAS_BORDER, width=3)
    # Canvas label
    draw.text((canvas_x + 5, canvas_y + 3), "Memory Canvas", font=font_bold(12), fill=CANVAS_BORDER)

    # Portrait inside canvas
    cp_x, cp_y = canvas_x + 15, canvas_y + 25
    draw_portrait_placeholder(img, cp_x, cp_y, 95, 125)

    # Text content beside portrait in canvas
    ct_x = cp_x + 110
    ct_y = cp_y + 5
    draw.text((ct_x, ct_y), "William Shakespeare", font=font_bold(12), fill=TEXT_COLOR)
    draw.text((ct_x, ct_y+18), "(1564 – 1616)", font=font(11), fill=DARK_GRAY)
    draw.text((ct_x, ct_y+38), "English playwright", font=font(11), fill=DARK_GRAY)
    draw.text((ct_x, ct_y+54), "and poet. Greatest", font=font(11), fill=DARK_GRAY)
    draw.text((ct_x, ct_y+70), "writer in English.", font=font(11), fill=DARK_GRAY)
    draw.text((ct_x, ct_y+92), "Works: Hamlet,", font=font(11), fill=DARK_GRAY)
    draw.text((ct_x, ct_y+108), "Romeo & Juliet...", font=font(11), fill=DARK_GRAY)

    # QA section inside canvas
    qa_y = cp_y + 138
    draw.line([(canvas_x+10, qa_y), (canvas_x+canvas_w-10, qa_y)], fill=(200, 210, 230), width=1)
    draw.text((canvas_x+15, qa_y+5), "Q: Who is this person?", font=font_bold(11), fill=BLUE)
    draw.text((canvas_x+15, qa_y+22), "A: Shakespeare, English playwright,", font=font(10), fill=DARK_GRAY)
    draw.text((canvas_x+15, qa_y+36), "   1564-1616, works include Hamlet...", font=font(10), fill=DARK_GRAY)

    # Visual highlight — "image preserved!"
    draw.rectangle([canvas_x-3, canvas_y-3, canvas_x+canvas_w+3, canvas_y+canvas_h+3],
                   outline=MEMCANVAS_ACCENT, width=2)

    # Annotation
    anno_rx = mid_x + 400
    draw.text((anno_rx, row2_top + 55), "Stored:", font=font_bold(15), fill=DARK_GRAY)
    draw.text((anno_rx, row2_top + 78), "✓  Text content", font=font(14), fill=(80, 130, 80))
    draw.text((anno_rx, row2_top + 100), "✓  Portrait image", font=font(14), fill=(80, 130, 80))
    draw.text((anno_rx, row2_top + 122), "✓  Visual details", font=font(14), fill=(80, 130, 80))
    draw.text((anno_rx, row2_top + 144), "✓  Spatial layout", font=font(14), fill=(80, 130, 80))

    draw.text((anno_rx, row2_top + 180), "All modalities unified", font=font_bold(14), fill=MEMCANVAS_ACCENT)
    draw.text((anno_rx, row2_top + 200), "on a single canvas image", font=font_bold(14), fill=MEMCANVAS_ACCENT)

    # ══════════════════════════════════════════════════════════
    # Arrows from Row 2 to Row 3
    # ══════════════════════════════════════════════════════════
    arrow2_y_start = row2_top + row2_h + 5
    arrow2_y_end = row2_top + row2_h + 42

    draw_arrow(draw, left_center, arrow2_y_start, left_center, arrow2_y_end, TEXTMEM_ACCENT, 3, 12)
    draw.text((left_center - 25, arrow2_y_start + 6), "recall", font=font(13), fill=TEXTMEM_ACCENT)

    draw_arrow(draw, right_center, arrow2_y_start, right_center, arrow2_y_end, MEMCANVAS_ACCENT, 3, 12)
    draw.text((right_center - 25, arrow2_y_start + 6), "recall", font=font(13), fill=MEMCANVAS_ACCENT)

    # ══════════════════════════════════════════════════════════
    # ROW 3: Later Query
    # ══════════════════════════════════════════════════════════
    row3_top = arrow2_y_end + 5
    row3_h = 300

    # User query (centered top)
    query_text = 'Later query:  "What color was Shakespeare\'s collar in that portrait?"'
    qf = font_bold(17)
    qw = draw.textbbox((0, 0), query_text, font=qf)[2]
    draw.text(((W - qw) // 2, row3_top + 5), query_text, font=qf, fill=TEXT_COLOR)

    # User icon
    draw_user_icon(draw, W//2 - qw//2 - 30, row3_top + 15, 16, USER_COLOR)

    row3_content_top = row3_top + 38

    # ── LEFT: Failure ──
    fail_box = (40, row3_content_top, mid_x - 25, row3_top + row3_h - 10)
    draw_rounded_rect(draw, fail_box, 14, fill=RED_BG, outline=RED, width=2)

    # Big X
    cross_cx, cross_cy = 110, row3_content_top + 55
    draw.ellipse([cross_cx-28, cross_cy-28, cross_cx+28, cross_cy+28], fill=RED)
    draw.text((cross_cx-14, cross_cy-18), "✗", font=font_bold(32), fill=WHITE)

    # Failure text
    fx = 160
    draw.text((fx, row3_content_top + 15), "Text Memory Response:", font=font_bold(15), fill=RED)
    draw.text((fx, row3_content_top + 42), '"I don\'t have visual information', font=font(14), fill=DARK_GRAY)
    draw.text((fx, row3_content_top + 62), ' about the portrait in my memory.', font=font(14), fill=DARK_GRAY)
    draw.text((fx, row3_content_top + 82), ' The stored summary only contains', font=font(14), fill=DARK_GRAY)
    draw.text((fx, row3_content_top + 102), ' text — no image was preserved."', font=font(14), fill=DARK_GRAY)

    draw.text((fx, row3_content_top + 135), "Failure: visual details lost", font=font_bold(14), fill=RED)
    draw.text((fx, row3_content_top + 155), "at memory storage time", font=font_bold(14), fill=RED)

    # ── RIGHT: Success ──
    succ_box = (mid_x + 25, row3_content_top, W - 40, row3_top + row3_h - 10)
    draw_rounded_rect(draw, succ_box, 14, fill=GREEN_BG, outline=GREEN, width=2)

    # Big checkmark
    check_cx, check_cy = mid_x + 100, row3_content_top + 55
    draw.ellipse([check_cx-28, check_cy-28, check_cx+28, check_cy+28], fill=GREEN)
    draw.text((check_cx-14, check_cy-18), "✓", font=font_bold(32), fill=WHITE)

    # Success text
    sx = mid_x + 150
    draw.text((sx, row3_content_top + 15), "MemCanvas Response:", font=font_bold(15), fill=GREEN)
    draw.text((sx, row3_content_top + 42), '"Based on the memory canvas,', font=font(14), fill=DARK_GRAY)
    draw.text((sx, row3_content_top + 62), ' Shakespeare is wearing a white', font=font(14), fill=DARK_GRAY)
    draw.text((sx, row3_content_top + 82), ' ruff collar in the portrait."', font=font(14), fill=DARK_GRAY)

    draw.text((sx, row3_content_top + 120), "Success: the canvas preserves", font=font_bold(14), fill=GREEN)
    draw.text((sx, row3_content_top + 140), "the original portrait image,", font=font_bold(14), fill=GREEN)
    draw.text((sx, row3_content_top + 160), "enabling visual recall", font=font_bold(14), fill=GREEN)

    # Mini canvas thumbnail in success area
    mini_x, mini_y = W - 195, row3_content_top + 45
    draw_rounded_rect(draw, (mini_x, mini_y, mini_x+120, mini_y+105), 6,
                      fill=WHITE, outline=CANVAS_BORDER, width=2)
    draw_portrait_placeholder(img, mini_x+8, mini_y+8, 55, 72)
    draw.text((mini_x+68, mini_y+15), "Shake-", font=font(9), fill=DARK_GRAY)
    draw.text((mini_x+68, mini_y+27), "speare", font=font(9), fill=DARK_GRAY)
    draw.text((mini_x+68, mini_y+42), "White", font=font_bold(9), fill=GREEN)
    draw.text((mini_x+68, mini_y+54), "ruff", font=font_bold(9), fill=GREEN)
    draw.text((mini_x+68, mini_y+66), "collar", font=font_bold(9), fill=GREEN)
    draw.text((mini_x+5, mini_y+85), "canvas memory", font=font(9), fill=CANVAS_BORDER)

    # ══════════════════════════════════════════════════════════
    # VS label
    # ══════════════════════════════════════════════════════════
    for vy in [row2_top + row2_h//2 - 15, row3_content_top + 60]:
        draw.ellipse([mid_x-18, vy, mid_x+18, vy+36], fill=(220, 225, 235), outline=MID_GRAY, width=2)
        draw.text((mid_x-12, vy+6), "VS", font=font_bold(14), fill=DARK_GRAY)

    return img


if __name__ == "__main__":
    img = create_figure()
    out_path = "/home/cyf/codex/paper_figures/fig1_motivation.png"
    img.save(out_path, dpi=(300, 300))
    print(f"Saved to {out_path}  ({img.size[0]}×{img.size[1]})")
