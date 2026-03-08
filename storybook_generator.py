import base64
import logging
import os
from google import genai
from google.genai.types import GenerateContentConfig, Modality

logger = logging.getLogger(__name__)

# Model for interleaved text + image generation
STORYBOOK_IMAGE_MODEL = os.getenv("STORYBOOK_IMAGE_MODEL", "gemini-2.5-flash-image")
# Fallback: text-only storybook when image model is unavailable
STORYBOOK_TEXT_MODEL = os.getenv("STORYBOOK_TEXT_MODEL", "gemini-2.5-flash")

STORYBOOK_PROMPT = """
You are writing a beautiful, intimate memory book — not a transcript, but a living story — 
capturing a video call between two loved ones separated by distance and language.

CALL DETAILS:
- Duration: {duration}
- People: {participants}
- Languages bridged: {languages}

THEIR WORDS (translated for you):
{voice_snippets}

SCENES FROM THEIR WORLD:
{scene_descriptions}

YOUR TASK:
Write a flowing, literary narrative in the style of a handwritten letter with photographs.
This is a memory they will treasure. Make it feel timeless.

NARRATIVE STRUCTURE — write approximately 5-6 "pages":
  Page 1 — "The Call Begins": Set the scene. Where are they? What time of day? 
            The moment the faces appear on screen.
  Page 2 — "First Words": The first things they say. The relief of seeing each other.
            Quote their actual words in italics.
  Page 3 — "Everyday Things": What they show each other. Food, flowers, a room.
            The beautiful ordinariness of love across distance.
  Page 4 — "What Wasn't Said": The emotion beneath the words.
            The years of longing in a simple "Have you eaten?"
  Page 5 — "Saying Goodbye": The hardest part. Hands pressed to glass.
            What they promise before hanging up.
  Page 6 — "What Remains": A reflection. What technology can bridge, what it cannot.
            The gift of having seen each other today.

STYLE RULES:
- Prose should be warm, literary, and unhurried — like Isabel Allende writing about family
- Quote their actual words in *italics* with attribution
- Each page ends with a single evocative sentence that lingers
- Languages used: weave in the original language phrases naturally (e.g., "她轻声说，'我想你了'")
- Length: each page ~3-4 paragraphs
- Keep emotional trajectory explicit: anticipation → relief → tenderness → ache → hope
- Use sensory specifics from scenes (light, hands, objects, weather, room sounds)
- Avoid vague summaries ("they felt emotional"); show emotion through concrete moments

{image_instruction}

Begin writing now. Start with Page 1.
"""

STORYBOOK_IMAGE_INSTRUCTION = """
IMAGES:
After each narrative page, generate ONE image in this style:
- Watercolor illustration, warm tones (amber, rose, sage)
- Intimate and soft — like a memory half-remembered
- Shows the scene described in that page
- NOT a photograph — a painted memory
"""

STORYBOOK_NO_IMAGE_INSTRUCTION = """
(Generate text narrative only — no images required.)
"""

async def generate_storybook(client: genai.Client, storybook_input: dict) -> list:
    """Generate an interactive storybook from call data.
    
    Tries image model first; falls back to text-only on rate limit or model error.
    """
    voice_snippets = storybook_input.get("voice_snippets", [])
    scene_descs = storybook_input.get("scene_descriptions", [])
    screenshots = storybook_input.get("screenshots", [])[:5]
    call_meta = storybook_input.get("call_metadata", {})

    voice_text = "\n".join([
        f'  • *"{s.get("text", "")}"* — {s.get("speaker", "?")} at {s.get("timestamp", "?")}' +
        (f'\n    (Translation: "{s.get("translation", "")}")' if s.get("translation") else "")
        for s in voice_snippets if isinstance(s, dict)
    ]) or "  (No voice moments captured — write from the scene descriptions.)"

    scene_text = "\n".join([
        f'  [{s.get("timestamp", "?")}] {s.get("description", "")}'
        for s in scene_descs if isinstance(s, dict)
    ]) or "  (No scene descriptions captured — imagine the setting from the voice moments.)"

    languages = call_meta.get("languages", [call_meta.get("lang_a", "?"), call_meta.get("lang_b", "?")])

    # Build the image input contents (actual screenshots from the call)
    img_contents = []
    for screenshot in screenshots:
        img_data = None
        mime_type = "image/jpeg"
        if isinstance(screenshot, dict):
            img_data = screenshot.get("data", "")
            mime_type = screenshot.get("mime_type", "image/jpeg")
        elif isinstance(screenshot, str):
            img_data = screenshot
        if img_data:
            if isinstance(img_data, bytes):
                img_data = base64.b64encode(img_data).decode()
            if img_data:
                img_contents.append({
                    "inline_data": {"mime_type": mime_type or "image/jpeg", "data": img_data}
                })

    if client is None:
        logger.warning("No Gemini client; using local storybook fallback")
        return _generate_local_storybook(voice_snippets, scene_descs, call_meta)

    # Try with image generation first
    pages = await _try_generate_with_images(
        client, img_contents, voice_text, scene_text, call_meta, languages
    )
    if pages:
        return pages

    # Fallback: text-only storybook (still beautiful)
    logger.info("Falling back to text-only storybook generation")
    pages = await _generate_text_only(
        client, img_contents, voice_text, scene_text, call_meta, languages
    )
    if pages:
        return pages

    logger.warning("Storybook model calls failed; using local storybook fallback")
    return _generate_local_storybook(voice_snippets, scene_descs, call_meta)


async def _build_prompt(voice_text: str, scene_text: str, call_meta: dict,
                         languages: list, with_images: bool) -> str:
    return STORYBOOK_PROMPT.format(
        duration=call_meta.get("duration", "an intimate call"),
        participants=" and ".join(call_meta.get("participants", ["two loved ones"])),
        languages=" ↔ ".join(languages) if languages else "multiple languages",
        voice_snippets=voice_text,
        scene_descriptions=scene_text,
        image_instruction=STORYBOOK_IMAGE_INSTRUCTION if with_images else STORYBOOK_NO_IMAGE_INSTRUCTION,
    )


async def _try_generate_with_images(
    client: genai.Client, img_contents: list,
    voice_text: str, scene_text: str, call_meta: dict, languages: list
) -> list:
    """Try to generate a storybook with interleaved images."""
    try:
        prompt = await _build_prompt(voice_text, scene_text, call_meta, languages, with_images=True)
        contents = img_contents + [{"text": prompt}]
        modalities = [Modality.TEXT, Modality.IMAGE]

        response = await client.aio.models.generate_content(
            model=STORYBOOK_IMAGE_MODEL,
            contents=contents,
            config=GenerateContentConfig(response_modalities=modalities),
        )
        pages = _parse_response(response)
        if pages:
            logger.info(f"Storybook generated with images: {len(pages)} parts")
        return pages
    except Exception as e:
        err_str = str(e)
        if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
            logger.warning(f"Image model rate-limited, falling back: {err_str[:100]}")
        elif "404" in err_str or "NOT_FOUND" in err_str:
            logger.warning(f"Image model unavailable, falling back: {err_str[:100]}")
        else:
            logger.error(f"Image storybook error: {type(e).__name__}: {err_str[:200]}")
        return []


async def _generate_text_only(
    client: genai.Client, img_contents: list,
    voice_text: str, scene_text: str, call_meta: dict, languages: list
) -> list:
    """Generate a text-only storybook — still emotionally rich."""
    try:
        prompt = await _build_prompt(voice_text, scene_text, call_meta, languages, with_images=False)
        contents = img_contents + [{"text": prompt}]

        response = await client.aio.models.generate_content(
            model=STORYBOOK_TEXT_MODEL,
            contents=contents,
            config=GenerateContentConfig(response_modalities=[Modality.TEXT]),
        )
        pages = _parse_response(response)
        logger.info(f"Text-only storybook generated: {len(pages)} parts")
        return pages
    except Exception as e:
        logger.error(f"Text-only storybook error: {type(e).__name__}: {e}")
        return []


def _parse_response(response) -> list:
    """Extract text and image parts from a Gemini response."""
    pages = []
    if not response:
        return pages
    if not response.candidates:
        if getattr(response, "text", None):
            text = response.text.strip()
            if text:
                pages.append({"type": "text", "content": text})
        return pages
    content = response.candidates[0].content
    if not content:
        if getattr(response, "text", None):
            text = response.text.strip()
            if text:
                pages.append({"type": "text", "content": text})
        return pages
    for part in content.parts:
        if hasattr(part, "text") and part.text and part.text.strip():
            pages.append({"type": "text", "content": part.text})
        elif hasattr(part, "inline_data") and part.inline_data:
            img = part.inline_data.data
            if isinstance(img, str):
                try:
                    img = base64.b64decode(img)
                except Exception:
                    pass
            pages.append({
                "type": "image",
                "content": img,
                "mime_type": part.inline_data.mime_type or "image/png",
            })
    return pages


def _generate_local_storybook(voice_snippets: list, scene_descs: list, call_meta: dict) -> list:
    """Deterministic fallback when model generation is unavailable."""
    participants = call_meta.get("participants", ["two loved ones"])
    if not isinstance(participants, list):
        participants = ["two loved ones"]
    pair = " and ".join([str(p) for p in participants[:2]]) or "two loved ones"
    duration = call_meta.get("duration", "a short call")
    languages = call_meta.get("languages", [])
    if not isinstance(languages, list):
        languages = []
    lang_text = " ↔ ".join([str(l) for l in languages if l]) or "different languages"

    cleaned_snippets = [
        s for s in voice_snippets
        if isinstance(s, dict) and (s.get("text") or s.get("translation"))
    ]
    cleaned_scenes = [
        s for s in scene_descs
        if isinstance(s, dict) and s.get("description")
    ]

    def _quote(idx: int, default: str) -> str:
        if idx < len(cleaned_snippets):
            s = cleaned_snippets[idx]
            q = s.get("translation") or s.get("text") or default
            who = s.get("speaker", "One of them")
            ts = s.get("timestamp", "sometime")
            return f'*"{q}"* — {who} ({ts})'
        return f'*"{default}"*'

    def _scene(idx: int, default: str) -> str:
        if idx < len(cleaned_scenes):
            s = cleaned_scenes[idx]
            return s.get("description", default)
        return default

    pages: list[dict] = []
    pages.append({
        "type": "text",
        "content": (
            "## Page 1 — The Call Begins\n"
            f"The connection flickers into place: {pair}, separated by geography, joined in real time through {lang_text}. "
            f"The call lasts {duration}, but in the first seconds the distance already starts to shrink.\n\n"
            f"{_scene(0, 'A familiar room, warm light, and a face that immediately feels like home.')}\n\n"
            "What begins as a call becomes a return."
        ),
    })
    pages.append({
        "type": "text",
        "content": (
            "## Page 2 — First Words\n"
            "Relief arrives as ordinary language: checking in, noticing details, confirming the other person is really there.\n\n"
            f"{_quote(0, 'I miss you.')}\n\n"
            "Even translated words can carry the exact weight of the original voice."
        ),
    })
    pages.append({
        "type": "text",
        "content": (
            "## Page 3 — Everyday Things\n"
            "They show each other small things: food, flowers, corners of a room, signs of life continuing in parallel.\n\n"
            f"{_scene(1, 'A table, a window, and everyday objects become proof of shared life.')}\n\n"
            f"{_quote(1, 'Have you eaten?')}\n\n"
            "Tenderness hides in practical questions."
        ),
    })
    pages.append({
        "type": "text",
        "content": (
            "## Page 4 — What Wasn’t Said\n"
            "Under the spoken lines lives the larger story: the missed meals, the quiet mornings, the years measured by screens and flights.\n\n"
            f"{_quote(2, 'I love you.')}\n\n"
            "Love often sounds simplest when it is most true."
        ),
    })
    pages.append({
        "type": "text",
        "content": (
            "## Page 5 — Saying Goodbye\n"
            "Goodbyes are careful and repetitive, as if repeating them can soften them.\n\n"
            f"{_scene(2, 'A hand lifting toward the camera, a pause before disconnecting.')}\n\n"
            "The call ends; the feeling does not."
        ),
    })
    pages.append({
        "type": "text",
        "content": (
            "## Page 6 — What Remains\n"
            "Technology carried sound, language, and image across distance. What it carried most faithfully was devotion.\n\n"
            f"{pair} return to separate rooms, but the same memory.\n\n"
            "They are apart, and somehow together."
        ),
    })
    return pages

def render_storybook_html(pages: list, title: str = "Our Moment Together",
                          stylized_images: list | None = None,
                          mood: str = "sentimental") -> str:
    """Render storybook pages to Polaroid-style, print-ready HTML.
    
    If stylized_images are provided (from background processing during the call),
    they are woven between text sections as Polaroid-framed photos.
    """
    subtitle = ""
    if " — " in title:
        parts = title.split(" — ", 1)
        title_main = parts[0]
        subtitle = parts[1]
    else:
        title_main = title

    if stylized_images is None:
        stylized_images = []

    img_count = sum(1 for p in pages if p.get("type") == "image") + len(stylized_images)
    page_count = sum(1 for p in pages if p.get("type") == "text")

    mood_bg = "#f7f3ee" if mood == "sentimental" else "#faf6f0"
    mood_accent = "#c4a888" if mood == "sentimental" else "#e8a54b"
    mood_label = "A love letter in images" if mood == "sentimental" else "The funny reel"

    html_parts = [f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title_main}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,500;1,400&family=Caveat:wght@400;600&family=DM+Sans:wght@300;400&display=swap" rel="stylesheet">
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'EB Garamond', Georgia, serif;
    background: {mood_bg};
    color: #2c2218;
    font-size: 19px;
    line-height: 1.85;
  }}

  .cover {{
    min-height: 55vh;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    padding: 4rem 2rem; text-align: center;
    background: linear-gradient(160deg, #f0e8dc 0%, #e8d8c8 50%, #f0e8dc 100%);
    border-bottom: 1px solid #d4c4b4;
    position: relative;
  }}
  .cover::before {{
    content: '在一起'; position: absolute;
    top: 2rem; left: 50%; transform: translateX(-50%);
    font-family: 'DM Sans', sans-serif; font-size: 0.75rem;
    letter-spacing: 0.3em; color: #a08870; text-transform: uppercase;
  }}
  .cover-title {{ font-size: clamp(2rem, 5vw, 3.2rem); font-weight: 400; color: #3a2a1a; line-height: 1.2; max-width: 600px; }}
  .cover-subtitle {{ font-family: 'DM Sans', sans-serif; font-size: 0.85rem; color: #8a7060; margin-top: 1.2rem; letter-spacing: 0.06em; }}
  .cover-divider {{ width: 60px; height: 1px; background: {mood_accent}; margin: 1.8rem auto; }}
  .cover-meta {{ font-size: 0.8rem; color: #a08070; font-family: 'DM Sans', sans-serif; letter-spacing: 0.05em; }}
  .cover-mood {{ display: inline-block; padding: 0.25rem 0.9rem; border: 1px solid {mood_accent}; border-radius: 16px; font-family: 'DM Sans', sans-serif; font-size: 0.72rem; color: {mood_accent}; margin-top: 0.8rem; letter-spacing: 0.08em; }}

  .story {{ max-width: 720px; margin: 0 auto; padding: 3rem 2rem 6rem; }}

  .page-section {{ margin: 3rem 0; page-break-inside: avoid; }}
  .page-section + .page-section {{ border-top: 1px solid #e0d4c8; padding-top: 3rem; }}

  .page-text {{ white-space: pre-wrap; word-wrap: break-word; }}
  .page-text p {{ margin-bottom: 1.2em; }}

  .page-header {{
    font-family: 'DM Sans', sans-serif; font-size: 0.72rem; font-weight: 400;
    letter-spacing: 0.2em; text-transform: uppercase; color: #a08070; margin-bottom: 1.2rem;
  }}
  .pull-quote {{
    font-style: italic; color: #6a4a3a; font-size: 1.05em;
    border-left: 2px solid {mood_accent}; padding: 0.3rem 0 0.3rem 1.4rem; margin: 1.6rem 0;
  }}

  /* ── Polaroid photo frame ── */
  .polaroid {{
    display: inline-block;
    background: #fff;
    padding: 12px 12px 40px 12px;
    box-shadow: 0 6px 24px rgba(40,20,0,0.12), 0 2px 4px rgba(40,20,0,0.06);
    margin: 2rem auto;
    transform: rotate(var(--rot, -1.5deg));
    transition: transform 0.3s;
    max-width: 88%;
    position: relative;
  }}
  .polaroid:nth-child(even) {{ --rot: 1.8deg; }}
  .polaroid:nth-child(3n) {{ --rot: -2.5deg; }}
  .polaroid:nth-child(4n) {{ --rot: 0.8deg; }}
  .polaroid:hover {{ transform: rotate(0deg) scale(1.02); }}
  .polaroid img {{
    display: block; width: 100%; border-radius: 1px;
  }}
  .polaroid-caption {{
    position: absolute; bottom: 8px; left: 16px; right: 16px;
    font-family: 'Caveat', cursive; font-size: 1rem;
    color: #5a4a3a; text-align: center;
  }}
  .polaroid-strip {{
    display: flex; flex-wrap: wrap; justify-content: center;
    gap: 1.5rem; margin: 2.5rem 0;
  }}
  .polaroid-strip .polaroid {{
    max-width: 280px; flex-shrink: 0;
  }}

  /* Model-generated image (from Gemini) */
  .page-image-wrap {{ margin: 2.5rem -1rem; text-align: center; }}
  .page-image {{ max-width: 100%; border-radius: 4px; box-shadow: 0 8px 32px rgba(60,30,10,0.12); }}

  .colophon {{
    text-align: center; padding: 3rem 0 2rem;
    border-top: 1px solid #e0d4c8; margin-top: 3rem;
    font-family: 'DM Sans', sans-serif; font-size: 0.75rem;
    color: #b0a090; letter-spacing: 0.1em;
  }}

  @media print {{ .cover {{ min-height: auto; }} body {{ font-size: 16px; }} }}
</style>
</head>
<body>

<div class="cover">
  <h1 class="cover-title">{title_main}</h1>
  {'<div class="cover-divider"></div><p class="cover-subtitle">' + subtitle + '</p>' if subtitle else '<div class="cover-divider"></div>'}
  <p class="cover-meta">{page_count} chapters · {img_count} Polaroids · created by 在一起</p>
  <div class="cover-mood">{mood_label}</div>
</div>

<div class="story">
"""]

    # Determine where to insert Polaroid images between text sections
    text_sections = [p for p in pages if p.get("type") == "text"]
    # Distribute stylized images evenly among text sections
    polaroid_slots: dict[int, list] = {}
    if stylized_images:
        n_text = max(len(text_sections), 1)
        for si, simg in enumerate(stylized_images):
            slot = min(si % n_text, n_text - 1)
            polaroid_slots.setdefault(slot, []).append(simg)

    image_index = 0
    text_idx = 0
    for i, page in enumerate(pages):
        if page["type"] == "text":
            raw_text = page["content"]

            lines = raw_text.split("\n")
            styled_lines = []
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("## ") or (stripped.startswith("**Page") and stripped.endswith("**")):
                    header_text = stripped.lstrip("#").strip().strip("*").strip()
                    styled_lines.append(f'<div class="page-header">{_escape(header_text)}</div>')
                elif stripped.startswith("**") and stripped.endswith("**") and len(stripped) < 80:
                    inner = stripped.strip("*").strip()
                    styled_lines.append(f'<div class="page-header">{_escape(inner)}</div>')
                elif stripped.startswith("*") and stripped.endswith("*") and not stripped.startswith("**"):
                    quote_text = stripped.strip("*").strip()
                    styled_lines.append(f'<div class="pull-quote">{_escape(quote_text)}</div>')
                elif stripped:
                    styled_lines.append(f'<p>{_escape(line)}</p>')

            html_parts.append('<div class="page-section"><div class="page-text">')
            html_parts.append("\n".join(styled_lines))
            html_parts.append("</div>")

            # Insert Polaroid images after this text section
            slot_images = polaroid_slots.get(text_idx, [])
            if slot_images:
                html_parts.append('<div class="polaroid-strip">')
                for simg in slot_images:
                    raw = simg.get("data", b"")
                    if isinstance(raw, bytes):
                        img_b64 = base64.b64encode(raw).decode()
                    elif isinstance(raw, str):
                        img_b64 = raw
                    else:
                        continue
                    mime = simg.get("mime_type", "image/png")
                    image_index += 1
                    caption = f"Moment #{simg.get('index', image_index)}"
                    html_parts.append(f"""<div class="polaroid">
  <img src="data:{mime};base64,{img_b64}" alt="Polaroid {image_index}">
  <div class="polaroid-caption">{_escape(caption)}</div>
</div>""")
                html_parts.append('</div>')

            html_parts.append("</div>")
            text_idx += 1

        elif page["type"] == "image":
            raw = page["content"]
            if isinstance(raw, bytes):
                img_data = base64.b64encode(raw).decode()
            elif isinstance(raw, str):
                img_data = raw
            else:
                continue
            mime = page.get("mime_type", "image/png")
            image_index += 1
            html_parts.append(f"""
<div class="page-image-wrap">
  <img class="page-image" src="data:{mime};base64,{img_data}" alt="Memory illustration {image_index}">
</div>""")

    # If there are stylized images that weren't inserted (no text sections), show them all
    if not text_sections and stylized_images:
        html_parts.append('<div class="polaroid-strip">')
        for simg in stylized_images:
            raw = simg.get("data", b"")
            if isinstance(raw, bytes):
                img_b64 = base64.b64encode(raw).decode()
            elif isinstance(raw, str):
                img_b64 = raw
            else:
                continue
            mime = simg.get("mime_type", "image/png")
            image_index += 1
            html_parts.append(f"""<div class="polaroid">
  <img src="data:{mime};base64,{img_b64}" alt="Polaroid {image_index}">
  <div class="polaroid-caption">Moment #{simg.get('index', image_index)}</div>
</div>""")
        html_parts.append('</div>')

    html_parts.append("""
<div class="colophon">
  &mdash; Generated by 在一起 — Together &mdash;<br>
  A call remembered. A story preserved.
</div>
</div>
</body>
</html>""")
    return "\n".join(html_parts)


def _escape(text: str) -> str:
    """Basic HTML escaping."""
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))
