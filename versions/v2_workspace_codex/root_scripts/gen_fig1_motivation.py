import base64
import json
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",
)

prompt = """Create an academic diagram on white background showing a comparison between text-only memory vs visual memory (MemCanvas).

Top: A chat where user shows a Shakespeare portrait (Renaissance painting, white ruff collar) and asks "Who is this person?"

Middle: Two memory storage paths side by side.
Left path "Text Memory": only stores text summary, no image.
Right path "MemCanvas": stores both portrait image and text on a canvas.

Bottom: User later asks "What color was Shakespeare's collar?"
Left gives wrong answer with red X. Right gives correct answer "white ruff collar" with green checkmark.

Style: clean academic figure, arrows connecting sections, soft blue-gray boxes."""

response = client.chat.completions.create(
    model="google/gemini-3.1-flash-image-preview",
    max_tokens=4096,
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ],
    extra_body={"modalities": ["image", "text"]}
)

output_path = "/home/cyf/codex/fig1_motivation.png"
saved = False

for choice in response.choices:
    msg = choice.message
    if hasattr(msg, 'content') and msg.content:
        if isinstance(msg.content, list):
            for part in msg.content:
                if hasattr(part, 'type'):
                    if part.type == 'image_url' and hasattr(part, 'image_url'):
                        url = part.image_url.url if hasattr(part.image_url, 'url') else part.image_url.get('url', '')
                        if url.startswith('data:'):
                            b64_data = url.split(',', 1)[1]
                            img_bytes = base64.b64decode(b64_data)
                            with open(output_path, 'wb') as f:
                                f.write(img_bytes)
                            print(f"Saved image to {output_path} ({len(img_bytes)} bytes)")
                            saved = True
                    elif part.type == 'text':
                        print(f"Text: {part.text[:200]}")
        elif isinstance(msg.content, str):
            print(f"String response: {msg.content[:300]}")

if not saved:
    print("No image found in response. Dumping raw response structure:")
    raw = response.model_dump()
    print(json.dumps(raw, indent=2, default=str)[:3000])
    with open("/home/cyf/codex/fig1_response_debug.json", 'w') as f:
        json.dump(raw, indent=2, default=str, fp=f)
    print("Saved debug response to fig1_response_debug.json")
