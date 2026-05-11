"""
Reproduce MemOCR-style Markdown-to-Image rendering.
Extracts core logic from MemOCR's markdown_api_server.py (arXiv:2601.21468).
"""

import markdown
import io
from PIL import Image
from playwright.sync_api import sync_playwright
import os

OUTPUT_DIR = "/home/cyf/codex/paper_canvas_examples/memocr_comparison"


def memocr_render(markdown_text: str, output_path: str):
    """
    Render markdown text to image using MemOCR's exact approach.
    Directly extracted from MemOCR md2img/markdown_api_server.py _markdown_to_image_sync().
    """
    # MemOCR preprocessing: strip backticks
    markdown_text = markdown_text.strip().strip("`")
    html_content = markdown.markdown(markdown_text, extensions=["extra"])

    # MemOCR's exact HTML template
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{
                background-color: white;
                margin: 2em;
                padding: 2em;
            }}
            .content {{
                padding: 0px;
                display: inline-block;
                white-space: pre-wrap;
            }}
            h1, h2, h3, h4, h5, h6, p, ul, ol, blockquote {{
                margin-top: 0.5em;
                margin-bottom: 0.5em;
                padding: 0;
            }}
            ul, ol {{
                margin-left: 20px;
            }}
            code {{
                background-color: #f4f4f4;
                padding: 2px 4px;
                border-radius: 3px;
                font-family: 'Courier New', monospace;
            }}
            pre {{
                background-color: #f4f4f4;
                padding: 10px;
                border-radius: 5px;
                overflow-x: auto;
            }}
        </style>
    </head>
    <body>
        <div class="content">
            {html_content}
        </div>
    </body>
    </html>
    """

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--disable-dev-shm-usage", "--disable-gpu", "--no-sandbox"],
        )
        page = browser.new_page()
        page.set_content(full_html)
        page.wait_for_load_state("domcontentloaded", timeout=5000)

        element_handle = page.locator(".content")
        box = element_handle.bounding_box()

        if box and box["width"] >= 10 and box["height"] >= 10:
            clip_area = {
                "x": box["x"],
                "y": box["y"],
                "width": box["width"],
                "height": box["height"],
            }
            screenshot_bytes = page.screenshot(
                clip=clip_area, omit_background=True, type="png"
            )
        else:
            screenshot_bytes = page.screenshot(
                full_page=True, omit_background=True, type="png"
            )

        page.close()
        browser.close()

    # Save
    img = Image.open(io.BytesIO(screenshot_bytes))
    img.save(output_path)
    print(f"  Saved: {output_path} ({img.width}x{img.height}, {len(screenshot_bytes)} bytes)")
    return img


# ===== Sample memories in MemOCR style =====
# MemOCR uses VLM-generated markdown summaries as memory entries.
# These examples mimic what MemOCR would store for different benchmarks.

SAMPLES = {
    # 1. ScienceQA-style memory (factual knowledge)
    "scienceqa_photosynthesis": """# Memory Update: Photosynthesis Process

## Key Information Extracted from Article

- **Photosynthesis** is the process by which plants convert light energy into chemical energy.
- It occurs primarily in the **chloroplasts** of plant cells.
- The overall equation: **6CO₂ + 6H₂O → C₆H₁₂O₆ + 6O₂**
- Light-dependent reactions occur in the **thylakoid membrane**.
- The Calvin cycle (light-independent reactions) occurs in the **stroma**.

## Previous Memory (Retained)

- Plants use sunlight to produce glucose and oxygen.
- Chlorophyll is the primary pigment involved.

## New Information Added

- The process requires both **water** and **carbon dioxide** as inputs.
- **ATP** and **NADPH** are produced during the light-dependent reactions.
- These energy carriers drive the Calvin cycle to fix carbon.

## Conclusion

Photosynthesis converts light energy to chemical energy through two stages: light-dependent reactions (in thylakoids) and the Calvin cycle (in stroma). The net products are glucose and oxygen.
""",

    # 2. HotpotQA-style memory (multi-hop reasoning)
    "hotpotqa_directors": """# Memory Update: Film Director Comparison

## Key Information Extracted

- **Christopher Nolan** directed *The Dark Knight* (2008), which grossed over $1 billion worldwide.
- **Denis Villeneuve** directed *Dune* (2021), adapted from Frank Herbert's 1965 novel.
- Both directors are known for large-scale science fiction films.
- Nolan was born in **London, England** in 1970.
- Villeneuve was born in **Gentilly, Quebec, Canada** in 1967.

## Previous Memory (Retained)

- Christopher Nolan directed Inception (2010) and Interstellar (2014).
- Denis Villeneuve directed Arrival (2016) and Blade Runner 2049 (2017).

## New Information Added

- Nolan's *Oppenheimer* (2023) won 7 Academy Awards including Best Picture.
- Villeneuve's *Dune: Part Two* (2024) was a critical and commercial success.
- Both directors prefer **practical effects** over excessive CGI.

## Conclusion

Christopher Nolan (British) and Denis Villeneuve (Canadian) are contemporary directors known for ambitious sci-fi films. Nolan is the older by birth year but Villeneuve started directing earlier. Both won major awards in 2023-2024.
""",

    # 3. OK-VQA-style memory (visual knowledge)
    "okvqa_landmarks": """# Memory Update: Famous Landmarks Identification

## Key Information Extracted from Image

- The image shows the **Eiffel Tower** in Paris, France.
- It was constructed in **1889** for the World's Fair (Exposition Universelle).
- The tower is **330 meters** tall (including antenna).
- Designed by engineer **Gustave Eiffel's** company.

## Previous Memory (Retained)

- The Eiffel Tower is located on the **Champ de Mars** near the Seine River.
- It is the most-visited paid monument in the world.

## New Information Added

- The tower was originally intended to be **temporary** (dismantled after 20 years).
- It was saved because of its usefulness as a **radio transmission tower**.
- The tower has **three levels** open to visitors.
- It is repainted every **7 years** using approximately 60 tons of paint.

## Conclusion

The Eiffel Tower, an iconic Parisian landmark built in 1889, stands 330m tall on the Champ de Mars. Originally temporary, it survived due to its radio utility.
""",

    # 4. InfographicVQA-style memory (structured data)
    "infovqa_revenue": """# Memory Update: Company Revenue Data 2023

## Key Information Extracted from Infographic

- **Apple Inc.** reported total revenue of **$383.3 billion** in FY2023.
- Revenue breakdown by segment:
  - iPhone: **$200.6B** (52.3%)
  - Services: **$85.2B** (22.2%)
  - Mac: **$29.4B** (7.7%)
  - iPad: **$28.3B** (7.4%)
  - Wearables: **$39.8B** (10.4%)

## Previous Memory (Retained)

- Apple's FY2022 revenue was $394.3 billion (slight YoY decline).
- Services has been the fastest-growing segment.

## New Information Added

- iPhone remains the **dominant revenue driver** at over 50%.
- Services revenue grew **9.1%** year-over-year.
- Overall revenue declined **2.8%** from FY2022.
- The company maintained a **gross margin of 44.1%**.

## Conclusion

Apple's FY2023 revenue was $383.3B, led by iPhone (52%) and Services (22%). Despite a slight overall decline, Services continues strong growth at +9.1% YoY.
""",

    # 5. LoCoMo-style memory (dialogue context)
    "locomo_conversation": """# Memory Update: Conversation with User about Travel Plans

## Key Information Extracted from Dialogue

- User is planning a trip to **Japan** in **April 2024**.
- They want to visit **Tokyo**, **Kyoto**, and **Osaka**.
- Budget is approximately **$3,000** for 10 days.
- User prefers **budget accommodations** (hostels, capsule hotels).
- They are interested in **cherry blossom viewing** (hanami).

## Previous Memory (Retained)

- User mentioned wanting to try **authentic ramen** in Tokyo.
- They have a **JR Pass** question from a previous conversation.

## New Information Added

- User decided on **April 1-10** travel dates.
- They want to attend a **tea ceremony** in Kyoto.
- User is **vegetarian**, needs restaurant recommendations.
- They prefer **public transportation** over taxis.

## Conclusion

User plans a 10-day Japan trip (Apr 1-10, 2024) visiting Tokyo→Kyoto→Osaka on a $3K budget. Key interests: cherry blossoms, tea ceremony, vegetarian ramen. Prefers budget stays and public transit.
""",

    # 6. Short memory (typical compact entry)
    "short_factual": """# Memory Update: Chemical Elements

## Key Facts

- **Gold (Au)** has atomic number **79** and is a **transition metal**.
- It has a melting point of **1,064°C**.
- Gold is one of the least reactive chemical elements.
- It is a good conductor of electricity and heat.

## Conclusion

Gold (Au, Z=79) is a dense, soft transition metal with high melting point (1064°C), excellent conductivity, and low reactivity.
""",
}


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== Reproducing MemOCR-style Markdown Rendering ===\n")

    for name, md_text in SAMPLES.items():
        out_path = os.path.join(OUTPUT_DIR, f"memocr_{name}.png")
        print(f"Rendering: {name}")
        memocr_render(md_text, out_path)

    print(f"\nDone! All images saved to {OUTPUT_DIR}/")
