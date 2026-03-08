# Voice Pipeline Integration — Deepgram STT + ElevenLabs TTS

## Architecture Overview

```
Participant A (Browser)          Server (FastAPI)           Participant B (Browser)
┌─────────────────────┐    ┌──────────────────────────┐    ┌─────────────────────┐
│  Mic → PCM 16kHz    │───▶│  Deepgram Streaming STT  │    │                     │
│                     │    │     ↓ transcript          │    │                     │
│                     │    │  Gemini 2.5 Flash         │    │                     │
│                     │    │   (translate + audio out) │───▶│  Audio playback     │
│                     │    │     ↓ also                │    │                     │
│  Caption (mine)  ◀──│────│  ElevenLabs TTS (fallback)│    │                     │
│                     │    │     ↓                     │    │                     │
│                     │    │  Transcript Logger         │    │                     │
│                     │    │   → room_transcript.md    │    │                     │
└─────────────────────┘    └──────────────────────────┘    └─────────────────────┘
```

## What Changed

### Backend (`server.py`)

1. **Deepgram Streaming STT** replaces Gradium STT as the primary transcription engine
   - WebSocket connection to `wss://api.deepgram.com/v1/listen`
   - Model: `nova-2` (best accuracy, supports turn detection)
   - Params: `encoding=linear16`, `sample_rate=16000`, `channels=1`, `interim_results=true`, `endpointing=300`, `utterance_end_ms=1500`
   - Per-participant Deepgram session — each speaker gets their own stream
   - Turn detection via `speech_final` + `utterance_end` events

2. **ElevenLabs TTS** replaces Gradium TTS as the fallback voice synthesizer
   - REST API: `POST https://api.elevenlabs.io/v1/text-to-speech/{voice_id}`
   - Model: `eleven_multilingual_v2`
   - Output format: `pcm_16000` (matches existing audio pipeline)
   - Voice: `JBFqnCBsd6RMkjVDRZzb` (George — warm, natural)

3. **Real-time Transcript Logger** saves conversation to markdown
   - File written to `transcripts/{room_id}.md` on every finalized utterance
   - Format designed for easy storybook consumption (see below)
   - Includes speaker, language, timestamp, and translated text
   - Auto-creates file on room creation, appends atomically

4. **Transcript Download Endpoint**
   - `GET /api/rooms/{room_id}/transcript` — returns raw markdown
   - `GET /api/rooms/{room_id}/transcript?format=json` — returns structured JSON

### Frontend (`participant.html`)

1. **Download Transcript Button** in the controls bar
   - Downloads `{room_id}_transcript.md` file directly
   - Available during and after the call

## Transcript Markdown Format

The transcript file is structured for easy consumption by the storybook generator:

```markdown
# Conversation Transcript — Room: {room_id}
**Date:** 2024-03-08
**Participants:** Steven (English), 奶奶 (Chinese)
**Languages:** English ↔ Chinese

---

## Conversation

**[00:00] Steven (English):**
Hey grandma, can you see me?

**[00:05] 奶奶 (Chinese):**
我看到你了！你看起来很好。

> *Translation: I can see you! You look great.*

**[00:12] Steven (English):**
I miss you so much. How's the garden?

**[00:18] 奶奶 (Chinese):**
花园很好，玫瑰花开了。

> *Translation: The garden is good, the roses bloomed.*

---

## Call Summary
- **Duration:** 05:23
- **Total exchanges:** 12
- **Key moments:** love, miss, garden, roses
```

## How Your Teammate Consumes the Transcript

The transcript markdown can be fed directly into `storybook_generator.py`:

```python
# Read the transcript
with open(f"transcripts/{room_id}.md", "r") as f:
    transcript_md = f.read()

# The storybook generator already accepts voice_snippets —
# the transcript provides richer, more complete data
```

Or via API:
```bash
# Download transcript during/after call
curl http://localhost:8000/api/rooms/demo/transcript > demo_transcript.md

# Get structured JSON for programmatic use
curl http://localhost:8000/api/rooms/demo/transcript?format=json
```

## Environment Variables

```env
# Required
GOOGLE_API_KEY=your_key

# Voice Pipeline (NEW)
DEEPGRAM_API_KEY=b336f8008143664bed0a0f8b6b456fd7efab1fc1
ELEVENLABS_API_KEY=sk_a4d0fb49b1f39726ee42907b3cdfa12919ca891d3cee85ee

# Optional (existing)
GRADIUM_API_KEY=your_key
```

## Dependencies Added

```
deepgram-sdk>=3.0.0
elevenlabs>=1.0.0
```

## Turn-Based Flow (Two People Talking)

```
1. Person A speaks → PCM audio streamed to server
2. Server pipes PCM to Deepgram STT (per-participant WebSocket)
3. Deepgram returns interim + final transcripts
4. Final transcript:
   a. Appended to room transcript markdown
   b. Sent to Person A as "mine" caption
   c. Fed to Gemini for translation + audio response
   d. Gemini translated audio → Person B speakers
   e. Gemini translation text → Person B caption ("theirs")
5. Person B speaks → same flow in reverse
6. Transcript file grows with each exchange
```

## Key Design Decisions

- **Deepgram runs server-side** (not browser-side) to keep API keys secure and reduce client complexity
- **Per-participant Deepgram sessions** so each person's speech is transcribed independently
- **Transcript saved incrementally** — no data loss if server crashes mid-call
- **Markdown format** chosen for human readability + easy parsing by storybook generator
- **ElevenLabs TTS is fallback-only** — Gemini Live handles primary audio translation; ElevenLabs kicks in when Gemini is rate-limited or for narration
