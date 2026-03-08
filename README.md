# 在一起 (Zài Yīqǐ) — Together

> A live AI agent that translates, narrates, and creates memories — so people separated by distance can truly be there with each other.

**Google NYC Build With AI Hackathon 2026**

---

## What It Does

在一起 is not a video call. It's not a translation app. It's **presence**.

Two people — speaking different languages, thousands of miles apart — have a natural conversation. No delays, no subtitles they have to read, no language barrier. They just *talk*, and they *understand*.

When the call ends, the system generates keepsakes: an **Interactive Storybook** (AI-generated illustrations woven with moments from the call) and a **Memory Video** (stylized images set to music).

### Features

| Feature | Description |
|---|---|
| **Live Translation** | Bidirectional speech translation via Gemini Live API |
| **Live Captions** | Translated text overlay in real time |
| **Presence Narration** | "Where are you?" triggers camera → AI describes the environment in the other person's language |
| **Visual Captions** | Scene analysis labels from Google Cloud Vision API |
| **Interactive Storybook** | Post-call: screenshots + voice snippets → Gemini Interleaved Output → illustrated storybook |
| **Memory Video** | Post-call: stylized images → Veo video keepsake |
| **Fallback Pipeline** | ElevenLabs STT/TTS if Gemini is rate-limited |
| **Billing Monitor** | Live cost tracking with configurable alerts at $3, $5, $9, $12, $15, $19, $21, $23, $24.9 |

---

## Tech Stack

| Component | Service |
|---|---|
| Real-time translation | Gemini Live API (`gemini-2.5-flash-native-audio-preview`) |
| Scene analysis | Google Cloud Vision API |
| Storybook generation | Gemini Interleaved Output (`gemini-2.5-flash-image`) |
| Image stylization | Gemini image generation |
| Video generation | Veo (`veo-2.0-generate-001`) |
| Backend | Python / FastAPI / WebSockets |
| Hosting | Google Cloud Run |
| Fallback STT/TTS | ElevenLabs Scribe + Flash |

---

## Quick Start

### Prerequisites

- Python 3.11+
- [Google AI Studio API key](https://aistudio.google.com/apikey)
- (Optional) Google Cloud project with Vision API enabled
- (Optional) ElevenLabs API key for fallback

### Setup

```bash
# Clone and enter
cd google-nyc-hackathon

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your keys

# Run
python server.py
```

Open [http://localhost:8000](http://localhost:8000) in two browser tabs. Join the same room with different languages. Start talking.

Demo URLs:
- `http://localhost:8000/demo/live` (live translation demo)
- `http://localhost:8000/demo/memory` (storybook + memory video demo)

### Deploy to Cloud Run

```bash
# Build and deploy
gcloud run deploy together \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars "GOOGLE_API_KEY=your_key" \
  --session-affinity \
  --port 8080
```

The `--session-affinity` flag is important for WebSocket support.

After deploy, use:
- `https://<your-service-url>/demo/live`
- `https://<your-service-url>/demo/memory`

---

## How It Works

```
Person A (Hindi)                    Person B (English)
     │                                    │
     ▼                                    ▼
[Mic] → Audio ─────────┬─────────── Audio ← [Mic]
                       │
                       ▼
              ┌─────────────────┐
              │  Cloud Run      │
              │  (FastAPI)      │
              │                 │
              │  Gemini Live    │◄── Bidirectional translation
              │  Vision API     │◄── Scene analysis (triggered)
              │  Screenshot     │◄── Capture for memorabilia
              └─────────────────┘
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
   Translated audio          Translated audio
   + Hindi captions          + English captions
```

**Post-call:** Screenshots, voice snippets, and scene descriptions feed into Gemini Interleaved Output (storybook) and Veo (memory video).

---

## Project Structure

```
├── server.py                  # FastAPI WebSocket server
├── storybook_generator.py     # Interactive Storybook pipeline
├── memory_video.py            # Memory Video pipeline
├── static/
│   ├── index.html             # Landing page
│   ├── participant.html       # Call UI
│   └── memories.html          # Post-call memorabilia page
├── requirements.txt
├── Dockerfile
└── .env.example
```

---

## Hackathon Categories

**Live Agents** — Bidirectional voice translation with barge-in support, real-time captions, and vision-triggered environment narration.

**Creative Storyteller** — Gemini Interleaved Output generates a flowing storybook with AI-illustrated pages from a single prompt, weaving screenshots, dialogue, and scene descriptions.

---

## The Close

> *在一起 means "together." That's what this is. Not a translation app. Not a video call. A way to actually be there — even when you can't.*
