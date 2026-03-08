#!/usr/bin/env python3
"""
在一起 — Demo Seeder
Seeds a room with an emotional grandmother/granddaughter call story.
The Gemini storybook will weave this into illustrated narrative pages.

Usage:
  python seed_demo.py                          # seeds against localhost:8080
  python seed_demo.py https://your-url.run.app # seeds against deployed Cloud Run
  python seed_demo.py http://localhost:8080 grandma-demo  # custom room name
"""
import sys
import json
import base64
import struct
import zlib
import urllib.request
import urllib.error

BASE_URL = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://localhost:8080"
ROOM_ID  = sys.argv[2] if len(sys.argv) > 2 else "grandma-demo"

# ─────────────────────────────────────────────────────────────────────────────
# The story: Nǎi Nai (grandmother, Beijing) calls Mei (granddaughter, New York)
# Nǎi Nai speaks Mandarin → 在一起 translates to English for Mei, and vice versa.
# ─────────────────────────────────────────────────────────────────────────────

PAYLOAD = {
    "lang_a": "Mandarin Chinese",
    "lang_b": "English",
    "participants": ["Nǎi Nai", "Mei"],

    # ── Voice moments captured during the call ──────────────────────────────
    "voice_snippets": [
        {
            "text": "我想你了，小梅。每天都想你。",
            "translation": "I miss you so much, little Mei. I think of you every single day.",
            "timestamp": "00:01",
            "speaker": "Nǎi Nai",
            "emotion": "longing"
        },
        {
            "text": "吃饭了吗？你在那边一定要好好吃饭。",
            "translation": "Have you eaten? You must take care of yourself over there.",
            "timestamp": "00:03",
            "speaker": "Nǎi Nai",
            "emotion": "warmth"
        },
        {
            "text": "Nai Nai, I'm eating your dumplings right now. I made them from the recipe you wrote me.",
            "translation": "奶奶，我正在吃你包的饺子。我用你写给我的食谱做的。",
            "timestamp": "00:05",
            "speaker": "Mei",
            "emotion": "joy"
        },
        {
            "text": "真的？你包的一定很好吃。你奶奶我教出来的！",
            "translation": "Really? They must taste wonderful. After all, I taught you!",
            "timestamp": "00:06",
            "speaker": "Nǎi Nai",
            "emotion": "pride"
        },
        {
            "text": "我给你看看我家门口的花，今年开得特别好。",
            "translation": "Let me show you the flowers by my door — they're especially beautiful this year.",
            "timestamp": "00:09",
            "speaker": "Nǎi Nai",
            "emotion": "joy"
        },
        {
            "text": "Oh, Nai Nai, they're gorgeous. Are those the ones from grandpa's garden?",
            "translation": "哇，奶奶，真漂亮。那是爷爷花园里的那些花吗？",
            "timestamp": "00:10",
            "speaker": "Mei",
            "emotion": "nostalgia"
        },
        {
            "text": "对，就是他种的那些。他要是看到你现在这么出息，一定很骄傲。",
            "translation": "Yes, the ones he planted. If he could see how well you've done, he'd be so proud.",
            "timestamp": "00:11",
            "speaker": "Nǎi Nai",
            "emotion": "bittersweet"
        },
        {
            "text": "I love you, Nai Nai. I'll come home soon.",
            "translation": "我爱你，奶奶。我很快就回家。",
            "timestamp": "00:14",
            "speaker": "Mei",
            "emotion": "love"
        },
        {
            "text": "我爱你，小梅。路上小心。奶奶在这里等你。",
            "translation": "I love you too, little Mei. Travel safe. Grandma will be right here waiting.",
            "timestamp": "00:15",
            "speaker": "Nǎi Nai",
            "emotion": "love"
        },
    ],

    # ── Scene descriptions from camera triggers ─────────────────────────────
    "scene_descriptions": [
        {
            "timestamp": "00:02",
            "description": (
                "An elderly woman's face glows warmly in a Beijing apartment. "
                "Faded red paper cuttings hang on the window behind her. "
                "A teapot steams on the table. Afternoon light filters through sheer curtains."
            ),
            "labels": ["person", "indoor", "warm lighting", "traditional decor"],
        },
        {
            "timestamp": "00:05",
            "description": (
                "A young woman holds up a plate of freshly made dumplings toward the camera in a New York studio apartment. "
                "She's smiling, flour still on her hands. "
                "A handwritten recipe card is visible on the counter beside her."
            ),
            "labels": ["person", "food", "cooking", "recipe", "joyful"],
        },
        {
            "timestamp": "00:09",
            "description": (
                "The grandmother points her phone at red and pink roses blooming beside a wooden gate. "
                "The gate opens onto a narrow Beijing hutong alley. "
                "Morning light catches the petals."
            ),
            "labels": ["outdoor", "flowers", "roses", "garden", "Beijing"],
        },
        {
            "timestamp": "00:12",
            "description": (
                "The grandmother holds up a framed photograph — a younger version of herself "
                "standing with a man in a garden. She's pointing to the flowers in the background, "
                "which match the roses she just showed."
            ),
            "labels": ["portrait", "memory", "family photo", "nostalgia"],
        },
        {
            "timestamp": "00:15",
            "description": (
                "Both screens in split view: Nǎi Nai pressing her palm to the camera glass in Beijing, "
                "and Mei pressing hers in New York. Two palms, one connection, six thousand miles apart."
            ),
            "labels": ["gesture", "connection", "emotional", "farewell"],
        },
    ],

    # ── Live captions from the translation session ──────────────────────────
    "captions": [
        {"text": "I miss you so much, little Mei. I think of you every single day.", "side": "theirs"},
        {"text": "我想你了，小梅。每天都想你。", "side": "mine"},
        {"text": "Have you eaten? You must take care of yourself over there.", "side": "theirs"},
        {"text": "Nai Nai, I'm eating your dumplings right now!", "side": "mine"},
        {"text": "我正在吃你包的饺子！", "side": "theirs"},
        {"text": "Really? They must taste wonderful. After all, I taught you!", "side": "theirs"},
        {"text": "真的？你包的一定很好吃。你奶奶我教出来的！", "side": "mine"},
        {"text": "Let me show you the flowers by my door — they're especially beautiful this year.", "side": "theirs"},
        {"text": "Oh Nai Nai, they're gorgeous. Are those from Grandpa's garden?", "side": "mine"},
        {"text": "Yes. If he could see how well you've done, he'd be so proud.", "side": "theirs"},
        {"text": "对，就是他种的那些。他要是看到你现在多出息，一定很骄傲。", "side": "mine"},
        {"text": "I love you, Nai Nai. I'll come home soon.", "side": "mine"},
        {"text": "我爱你，小梅。路上小心。奶奶在这里等你。", "side": "theirs"},
        {"text": "I love you too, little Mei. Grandma will be right here waiting.", "side": "theirs"},
    ],
}


def _png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    return (
        struct.pack("!I", len(payload))
        + chunk_type
        + payload
        + struct.pack("!I", zlib.crc32(chunk_type + payload) & 0xFFFFFFFF)
    )


def _make_gradient_png_b64(width: int, height: int, top_rgb: tuple[int, int, int], bottom_rgb: tuple[int, int, int]) -> str:
    rows = bytearray()
    den = max(1, height - 1)
    for y in range(height):
        t = y / den
        r = int(top_rgb[0] * (1 - t) + bottom_rgb[0] * t)
        g = int(top_rgb[1] * (1 - t) + bottom_rgb[1] * t)
        b = int(top_rgb[2] * (1 - t) + bottom_rgb[2] * t)
        rows.append(0)
        rows.extend(bytes((r, g, b)) * width)
    ihdr = struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(bytes(rows), 9))
        + _png_chunk(b"IEND", b"")
    )
    return base64.b64encode(png).decode("ascii")


PAYLOAD["screenshots"] = [
    {
        "data": _make_gradient_png_b64(640, 360, (35, 24, 28), (152, 108, 84)),
        "mime_type": "image/png",
        "timestamp": "00:02",
        "description": "Warm apartment light and an intimate face on screen.",
    },
    {
        "data": _make_gradient_png_b64(640, 360, (28, 40, 58), (194, 146, 112)),
        "mime_type": "image/png",
        "timestamp": "00:09",
        "description": "Flowers and alleyway tones, bright and nostalgic.",
    },
    {
        "data": _make_gradient_png_b64(640, 360, (22, 20, 30), (132, 86, 120)),
        "mime_type": "image/png",
        "timestamp": "00:15",
        "description": "Two hands lifted to camera glass, two cities feeling close.",
    },
]
PAYLOAD["stylized_images"] = [
    {"data": s["data"], "mime_type": s["mime_type"], "index": i + 1, "mood": "sentimental"}
    for i, s in enumerate(PAYLOAD["screenshots"][:2])
]

def seed(base_url: str, room_id: str):
    url = f"{base_url}/api/rooms/{room_id}/seed"
    data = json.dumps(PAYLOAD).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read())
            print(f"\n✅  Seeded room '{room_id}'")
            print(f"   Screenshots   : {result.get('screenshots', 0)}")
            print(f"   Voice moments : {result.get('voice_snippets', 0)}")
            print(f"   Scene descriptions: {result.get('scene_descriptions', 0)}")
            print(f"   Captions      : {result.get('captions', 0)}")
            print(f"\n🎬  Open the memories page:")
            print(f"   {base_url}/room/{room_id}/memories")
            print(f"\n   The storybook will generate illustrated pages from Nǎi Nai and Mei's call.")
    except urllib.error.HTTPError as e:
        print(f"❌  HTTP {e.code}: {e.read().decode()}")
    except Exception as e:
        print(f"❌  Error: {e}")
        print(f"   Is the server running at {base_url}?")

if __name__ == "__main__":
    print(f"→ Seeding demo room '{ROOM_ID}' at {BASE_URL}…")
    seed(BASE_URL, ROOM_ID)
