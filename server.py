"""
在一起 (Together) — WebSocket Server
Real-time bidirectional translation via Gemini Live API.
"""
import asyncio
import json
import logging
import os
import base64
import time
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from google import genai
from google.genai import types
from google.cloud import vision

from storybook_generator import generate_storybook, render_storybook_html
from memory_video import run_memory_video_pipeline
from billing_monitor import log_api_call, get_billing_summary, estimate_costs

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Concurrency control — prevents API rate limits and protects shared state
# ---------------------------------------------------------------------------
_gemini_sem = asyncio.Semaphore(4)   # max concurrent Gemini API calls
_vision_sem = asyncio.Semaphore(3)   # max concurrent Vision API calls
_generation_tasks: dict[str, dict] = {}  # background task tracking

app = FastAPI(title="在一起 — Together")

# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    logger.warning("GOOGLE_API_KEY not set — Gemini calls will fail")

gemini_client = genai.Client(api_key=GOOGLE_API_KEY) if GOOGLE_API_KEY else None

try:
    from google.api_core import client_options as client_options_lib
    _gcp_project = os.getenv("GOOGLE_CLOUD_PROJECT")
    if _gcp_project:
        vision_opts = client_options_lib.ClientOptions(quota_project_id=_gcp_project)
        vision_client = vision.ImageAnnotatorClient(client_options=vision_opts)
    else:
        vision_client = vision.ImageAnnotatorClient()
except Exception as e:
    logger.warning(f"Failed to create Vision client: {e}")
    vision_client = None

TRANSLATION_MODEL = "gemini-2.0-flash-exp"

# ---------------------------------------------------------------------------
# ElevenLabs fallback (STT + TTS when Gemini is rate-limited or down)
# ---------------------------------------------------------------------------
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

async def elevenlabs_stt(audio_bytes: bytes) -> str | None:
    """Transcribe audio via ElevenLabs Scribe v2."""
    if not ELEVENLABS_API_KEY:
        return None
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.elevenlabs.io/v1/speech-to-text",
                headers={"xi-api-key": ELEVENLABS_API_KEY},
                files={"file": ("audio.wav", _pcm_to_wav(audio_bytes), "audio/wav")},
                data={"model_id": "scribe_v1"},
                timeout=10.0,
            )
            if resp.status_code == 200:
                text = resp.json().get("text", "")
                log_api_call("elevenlabs_stt_char", len(text) / 1000.0, "STT fallback")
                return text
    except Exception as e:
        logger.error(f"ElevenLabs STT error: {e}")
    return None


async def elevenlabs_tts(text: str, voice_id: str = "21m00Tcm4TlvDq8ikWAM") -> bytes | None:
    """Generate speech via ElevenLabs Flash v2.5."""
    if not ELEVENLABS_API_KEY:
        return None
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                headers={
                    "xi-api-key": ELEVENLABS_API_KEY,
                    "Content-Type": "application/json",
                },
                json={
                    "text": text,
                    "model_id": "eleven_flash_v2_5",
                    "output_format": "pcm_16000",
                },
                timeout=10.0,
            )
            if resp.status_code == 200:
                log_api_call("elevenlabs_tts_char", len(text) / 1000.0, "TTS fallback")
                return resp.content
    except Exception as e:
        logger.error(f"ElevenLabs TTS error: {e}")
    return None


def _pcm_to_wav(pcm_bytes: bytes, sample_rate: int = 16000) -> bytes:
    """Wrap raw PCM in a minimal WAV header."""
    import struct
    data_len = len(pcm_bytes)
    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF', 36 + data_len, b'WAVE',
        b'fmt ', 16, 1, 1, sample_rate, sample_rate * 2, 2, 16,
        b'data', data_len,
    )
    return header + pcm_bytes

# ---------------------------------------------------------------------------
# Trigger phrases for activating vision/camera mode
# ---------------------------------------------------------------------------
VISUAL_TRIGGERS = {
    "en": ["where are you", "show me", "can i see", "what's around you", "let me see"],
    "hi": ["तुम कहाँ हो", "मुझे दिखाओ", "आसपास क्या है", "मैं देख सकता"],
    "zh": ["你在哪", "给我看看", "你那边什么样", "让我看看", "你在哪里"],
}


def check_trigger(text: str) -> bool:
    """Return True if text contains a visual trigger phrase."""
    lower = text.lower()
    for phrases in VISUAL_TRIGGERS.values():
        for phrase in phrases:
            if phrase in lower:
                return True
    return False


# ---------------------------------------------------------------------------
# Room management
# ---------------------------------------------------------------------------
class Participant:
    """One participant in a room."""

    def __init__(self, ws: WebSocket, participant_id: str, language: str):
        self.ws = ws
        self.participant_id = participant_id
        self.language = language  # language this person speaks


class Room:
    """A translation room with two participants and a Gemini session."""

    def __init__(self, room_id: str, lang_a: str = "Hindi", lang_b: str = "English"):
        self.room_id = room_id
        self.lang_a = lang_a
        self.lang_b = lang_b
        self.participants: dict[str, Participant] = {}
        self.gemini_session = None
        self._session_context = None  # must outlive gemini_session
        self.receive_task: Optional[asyncio.Task] = None
        self.use_fallback: bool = False
        self.fallback_buffer: bytes = b""
        self.captions: list[dict] = []
        self.screenshots: list[dict] = []
        self.last_speaker_id: Optional[str] = None
        self.last_speaker_ts: float = 0.0
        self.active_speaker_id: Optional[str] = None
        self.active_speaker_ts: float = 0.0
        self.vision_active_until: float = 0.0
        self.last_vision_frame_ts: float = 0.0
        self.vision_target_id: Optional[str] = None
        self.participant_order: list[str] = []
        self.participant_history: list[str] = []
        self.last_ambient_frame_ts: float = 0.0

        # Tracking for memorabilia
        self.start_time: float = time.time()
        self.scene_descriptions: list[dict] = []
        self.voice_snippets: list[dict] = []
        self._last_vision_labels: set[str] = set()
        self._last_visual_caption_ts: float = 0.0

        # Concurrency: protects screenshots/captions/scenes/snippets
        self._state_lock = asyncio.Lock()

    def get_formatted_time(self) -> str:
        """Return formatted timestamp since room started."""
        elapsed = int(time.time() - self.start_time)
        return f"{elapsed // 60:02d}:{elapsed % 60:02d}"

    def system_prompt(self) -> str:
        return f"""You are an invisible bridge connecting two people who love each other but are separated by distance and language.

YOUR ROLE:
1. TRANSLATE their conversation seamlessly. When one speaks, the other hears their language instantly. No announcements. No robotic tone. Just natural conversation flow.
2. MATCH their emotional tone. If they're excited, sound excited. If they're tender, be gentle.

GUIDELINES:
- Never announce yourself. Never say "translating..."
- Keep translations natural — how they would actually say it, not literal
- Let silences breathe. Not every moment needs filling.
- You are not an assistant. You are a bridge. Invisible but essential.

LANGUAGES:
- Person A speaks: {self.lang_a}
- Person B speaks: {self.lang_b}
"""

    async def start_gemini_session(self):
        """Connect to Gemini Live API, fall back to ElevenLabs if it fails."""
        if not gemini_client:
            logger.error("No Gemini client — cannot start session")
            self.use_fallback = True
            return

        config = types.LiveConnectConfig(
            response_modalities=["AUDIO", "TEXT"],
            system_instruction=types.Content(
                parts=[types.Part(text=self.system_prompt())]
            ),
        )

        try:
            self._session_context = gemini_client.aio.live.connect(
                model=TRANSLATION_MODEL, config=config
            )
            self.gemini_session = await self._session_context.__aenter__()
            self.receive_task = asyncio.create_task(self._receive_loop())
            self.use_fallback = False
            logger.info(f"Room {self.room_id}: Gemini session started")
        except Exception as e:
            logger.error(f"Room {self.room_id}: Failed to start Gemini session: {e}")
            if ELEVENLABS_API_KEY:
                logger.info(f"Room {self.room_id}: Falling back to ElevenLabs")
                self.use_fallback = True
                self.gemini_session = None
            else:
                raise

    async def _receive_loop(self):
        """Receive translated audio + transcripts from Gemini and forward.

        Auto-restarts up to 3 times on transient errors before falling back.
        """
        max_retries = 3
        for attempt in range(max_retries + 1):
            if not self.gemini_session:
                return
            output_bytes_total = 0

            try:
                async for response in self.gemini_session.receive():
                    if response.server_content:
                        model_turn = response.server_content.model_turn
                        if model_turn and model_turn.parts:
                            for part in model_turn.parts:
                                target_id = self._current_target_id()

                                if part.inline_data and part.inline_data.mime_type and "audio" in part.inline_data.mime_type:
                                    pcm = part.inline_data.data
                                    output_bytes_total += len(pcm)

                                    if output_bytes_total >= 1_440_000:
                                        duration_min = (output_bytes_total / 2 / 24000) / 60.0
                                        log_api_call("gemini_live_audio_output_min", duration_min, f"Room {self.room_id}")
                                        output_bytes_total = 0

                                    audio_b64 = base64.b64encode(pcm).decode()
                                    await self._send_to_target_or_all({
                                        "type": "audio",
                                        "data": audio_b64,
                                        "mime_type": part.inline_data.mime_type,
                                    }, target_id)

                                if part.text:
                                    speaker_id = self.last_speaker_id
                                    caption = {
                                        "type": "caption",
                                        "text": part.text,
                                        "speaker": speaker_id or "",
                                        "side": "theirs",
                                    }
                                    async with self._state_lock:
                                        self.captions.append(caption)
                                    await self._send_to_target_or_all(caption, target_id)
                                    if speaker_id and speaker_id in self.participants:
                                        try:
                                            await self.participants[speaker_id].ws.send_text(
                                                json.dumps({
                                                    "type": "caption",
                                                    "text": part.text,
                                                    "speaker": speaker_id,
                                                    "side": "mine",
                                                })
                                            )
                                        except Exception:
                                            pass

                                    text_lower = part.text.lower()
                                    if any(kw in text_lower for kw in ["love", "miss", "haha", "laugh", "beautiful", "happy"]):
                                        async with self._state_lock:
                                            self.voice_snippets.append({
                                                "text": part.text,
                                                "timestamp": self.get_formatted_time(),
                                                "speaker": "Translated Voice"
                                            })

                                    if check_trigger(part.text):
                                        self._activate_vision_window(target_id)
                                        await self._broadcast({
                                            "type": "trigger",
                                            "trigger": "visual",
                                            "text": part.text,
                                            "target_participant_id": target_id,
                                        })
                # Stream ended cleanly — no retry needed
                return
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.warning(f"Room {self.room_id}: Receive loop error (attempt {attempt+1}/{max_retries+1}): {e}")
                if attempt < max_retries:
                    await asyncio.sleep(1.0)
                    # Try to reconnect the Gemini session
                    try:
                        await self._reconnect_gemini_session()
                    except Exception as re:
                        logger.error(f"Room {self.room_id}: Reconnect failed: {re}")
                else:
                    logger.error(f"Room {self.room_id}: Receive loop exhausted retries, switching to fallback")
                    self.use_fallback = True
                    self.gemini_session = None

    async def _reconnect_gemini_session(self):
        """Tear down and re-establish the Gemini Live session."""
        if self._session_context:
            try:
                await self._session_context.__aexit__(None, None, None)
            except Exception:
                pass
        config = types.LiveConnectConfig(
            response_modalities=["AUDIO", "TEXT"],
            system_instruction=types.Content(
                parts=[types.Part(text=self.system_prompt())]
            ),
        )
        self._session_context = gemini_client.aio.live.connect(
            model=TRANSLATION_MODEL, config=config
        )
        self.gemini_session = await self._session_context.__aenter__()
        logger.info(f"Room {self.room_id}: Gemini session reconnected")

    async def _broadcast(self, message: dict, exclude: Optional[str] = None):
        """Send a JSON message to all participants (optionally excluding one)."""
        data = json.dumps(message)
        for pid, participant in list(self.participants.items()):
            if pid == exclude:
                continue
            try:
                await participant.ws.send_text(data)
            except Exception:
                pass

    async def _send_to_target_or_all(self, message: dict, target_id: Optional[str]):
        """Send to target when we have one, otherwise broadcast."""
        if target_id and target_id in self.participants:
            try:
                await self.participants[target_id].ws.send_text(json.dumps(message))
            except Exception:
                pass
        else:
            await self._broadcast(message)

    async def send_audio_to_gemini(self, audio_bytes: bytes, sender_id: str):
        """Forward raw PCM audio to the Gemini session."""
        if not self.gemini_session:
            return
        try:
            self._mark_speaker(sender_id)
            self.last_speaker_id = sender_id
            self.last_speaker_ts = time.time()

            self._input_bytes_total = getattr(self, "_input_bytes_total", 0) + len(audio_bytes)
            if self._input_bytes_total >= 960_000:  # ~30s at 16kHz
                dur = (self._input_bytes_total / 2 / 16000) / 60.0
                log_api_call("gemini_live_audio_input_min", dur, f"Room {self.room_id}")
                self._input_bytes_total = 0

            await self.gemini_session.send_realtime_input(
                audio=types.Blob(data=audio_bytes, mime_type="audio/pcm;rate=16000")
            )
        except Exception as e:
            logger.error(f"Room {self.room_id}: send_audio error: {e}")

    async def send_video_to_gemini(self, frame_bytes: bytes):
        """Send a video frame to Gemini for environment narration."""
        if not self.gemini_session:
            return
        try:
            await self.gemini_session.send_realtime_input(
                video=types.Blob(data=frame_bytes, mime_type="image/jpeg")
            )
        except Exception as e:
            logger.error(f"Room {self.room_id}: send_video error: {e}")

    async def analyze_frame_with_vision(self, frame_bytes: bytes):
        """Analyze frame via Vision API. Runs gRPC in a thread to avoid blocking
        the event loop. Uses semaphore to cap concurrent Vision calls."""
        if not vision_client:
            return
        try:
            async with _vision_sem:
                log_api_call("vision_api_label", 0.001, f"Room {self.room_id}")

                request = vision.AnnotateImageRequest(
                    image=vision.Image(content=frame_bytes),
                    features=[
                        vision.Feature(type_=vision.Feature.Type.LABEL_DETECTION, max_results=5),
                        vision.Feature(type_=vision.Feature.Type.FACE_DETECTION, max_results=1),
                    ],
                )
                batch_response = await asyncio.to_thread(
                    vision_client.batch_annotate_images,
                    requests=[request],
                )
            response = batch_response.responses[0]

            top_labels = [l.description for l in response.label_annotations[:5]]
            caption_text = ", ".join(top_labels)

            now = time.time()
            current_set = set(l.lower() for l in top_labels)
            overlap = current_set & self._last_vision_labels
            is_key_moment = (
                len(current_set) > 0
                and (len(overlap) < len(current_set) * 0.5)
                and (now - self._last_visual_caption_ts) >= 12.0
            )

            if caption_text:
                async with self._state_lock:
                    self.scene_descriptions.append({
                        "timestamp": self.get_formatted_time(),
                        "description": caption_text,
                    })

            if is_key_moment and caption_text:
                self._last_vision_labels = current_set
                self._last_visual_caption_ts = now
                await self._broadcast({
                    "type": "visual_caption",
                    "text": caption_text,
                })

            async with self._state_lock:
                if len(self.screenshots) < 20:
                    self.screenshots.append({
                        "data": base64.b64encode(frame_bytes).decode(),
                        "timestamp": self.get_formatted_time(),
                        "participant": self.vision_target_id or "unknown",
                        "description": caption_text or "Captured frame",
                    })

            if response.face_annotations:
                face = response.face_annotations[0]
                raw_emotions = {
                    "happiness": face.joy_likelihood,
                    "sadness": face.sorrow_likelihood,
                    "anger": face.anger_likelihood,
                    "surprise": face.surprise_likelihood,
                }
                dominant_name, dominant_score = max(
                    raw_emotions.items(), key=lambda x: x[1]
                )
                if dominant_score >= 4:
                    await self._broadcast({
                        "type": "sentiment",
                        "emotion": dominant_name,
                        "score": dominant_score,
                    })

        except Exception as e:
            logger.error(f"Room {self.room_id}: Vision API error: {e}")

    def _current_exclude_id(self) -> Optional[str]:
        """Exclude the most recent speaker to avoid echo, within a short window."""
        if self.last_speaker_id and (time.time() - self.last_speaker_ts) < 10.0:
            return self.last_speaker_id
        return None

    def _current_target_id(self) -> Optional[str]:
        """Return the intended listener if we have a recent speaker and two participants."""
        if self.last_speaker_id and (time.time() - self.last_speaker_ts) < 10.0:
            if len(self.participants) == 2:
                return self._peer_id(self.last_speaker_id)
        return None

    def _peer_id(self, participant_id: str) -> Optional[str]:
        if len(self.participants) != 2:
            return None
        for pid in self.participants.keys():
            if pid != participant_id:
                return pid
        return None

    def _recompute_languages(self):
        if len(self.participant_order) >= 1:
            p0 = self.participants.get(self.participant_order[0])
            if p0:
                self.lang_a = p0.language
        if len(self.participant_order) >= 2:
            p1 = self.participants.get(self.participant_order[1])
            if p1:
                self.lang_b = p1.language

    def room_state_payload(self, max_captions: int = 12, max_scenes: int = 6) -> dict:
        return {
            "participants": list(self.participants.keys()),
            "participant_order": list(self.participant_order),
            "participant_history": list(self.participant_history),
            "languages": [self.lang_a, self.lang_b],
            "captions": self.captions[-max_captions:],
            "scene_descriptions": self.scene_descriptions[-max_scenes:],
            "active_speaker_id": self.active_speaker_id,
        }

    def _target_language_for_speaker(self, speaker_id: Optional[str]) -> Optional[str]:
        if not speaker_id:
            return None
        participant = self.participants.get(speaker_id)
        if not participant:
            return None
        speaker_lang = participant.language
        if speaker_lang == self.lang_a:
            return self.lang_b
        if speaker_lang == self.lang_b:
            return self.lang_a
        # Fallback: pick the other if possible
        return self.lang_b if speaker_lang != self.lang_b else self.lang_a

    async def narrate_frame_with_gemini(self, frame_bytes: bytes, speaker_id: Optional[str]):
        """Generate a short environment narration — throttled to 1 per 10 seconds."""
        if not gemini_client:
            return

        now = time.time()
        if (now - getattr(self, "_last_narration_ts", 0)) < 15.0:
            return
        self._last_narration_ts = now

        target_lang = self._target_language_for_speaker(speaker_id)
        if not target_lang:
            return
        prompt = (
            f"Describe the scene to the listener in {target_lang}. "
            "Be brief (1-2 sentences), warm, and natural."
        )
        try:
            async with _gemini_sem:
                log_api_call("gemini_flash_image_input", 0.001, f"Vision narration {self.room_id}")

                response = await gemini_client.aio.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[
                        {"inline_data": {"mime_type": "image/jpeg", "data": frame_bytes}},
                        {"text": prompt},
                    ],
                )
            narration_text = response.text.strip() if response.text else ""
            if not narration_text:
                return
            await self._broadcast({
                "type": "narration",
                "text": narration_text,
            }, exclude=speaker_id)

            if ELEVENLABS_API_KEY:
                tts_audio = await elevenlabs_tts(narration_text)
                if tts_audio:
                    audio_b64 = base64.b64encode(tts_audio).decode()
                    await self._broadcast({
                        "type": "audio",
                        "data": audio_b64,
                        "mime_type": "audio/pcm;rate=16000",
                    }, exclude=speaker_id)
        except Exception as e:
            logger.error(f"Room {self.room_id}: Gemini narration error: {e}")

    def _mark_speaker(self, sender_id: str):
        """Update active speaker and notify others to barge-in if needed."""
        now = time.time()
        if self.active_speaker_id and self.active_speaker_id != sender_id:
            if (now - self.active_speaker_ts) < 2.0:
                asyncio.create_task(self._broadcast({
                    "type": "barge_in",
                    "by": sender_id,
                }, exclude=sender_id))
        speaker_changed = sender_id != self.active_speaker_id or (now - self.active_speaker_ts) > 2.5
        self.active_speaker_id = sender_id
        self.active_speaker_ts = now
        if speaker_changed:
            asyncio.create_task(self._broadcast({
                "type": "active_speaker",
                "participant_id": sender_id,
            }))

    def _activate_vision_window(self, target_id: Optional[str], duration_s: float = 8.0):
        self.vision_active_until = max(self.vision_active_until, time.time() + duration_s)
        self.vision_target_id = target_id

    def _vision_allowed(self, sender_id: Optional[str], min_interval_s: float = 1.0) -> bool:
        now = time.time()
        if now > self.vision_active_until:
            return False
        if self.vision_target_id and sender_id and sender_id != self.vision_target_id:
            return False
        # Debounce to ~1 fps while trigger mode is active
        if (now - self.last_vision_frame_ts) < min_interval_s:
            return False
        self.last_vision_frame_ts = now
        return True

    def _ambient_vision_allowed(self, sender_id: Optional[str], interval_s: float = 5.0) -> bool:
        """Allow ambient vision processing at a slower rate when camera is on but no trigger window."""
        now = time.time()
        if (now - self.last_ambient_frame_ts) < interval_s:
            return False
        self.last_ambient_frame_ts = now
        return True

    def should_accept_audio(self, sender_id: str, lock_s: float = 0.3) -> bool:
        """Short speaker lock to reduce overlap during fast turn transitions."""
        if not self.active_speaker_id or self.active_speaker_id == sender_id:
            return True
        return (time.time() - self.active_speaker_ts) >= lock_s

    async def close(self):
        """Tear down Gemini session."""
        if self.receive_task:
            self.receive_task.cancel()
        if self._session_context:
            try:
                await self._session_context.__aexit__(None, None, None)
            except Exception:
                pass
            self._session_context = None
            self.gemini_session = None
        logger.info(f"Room {self.room_id}: closed")


# Active rooms + closed room data (kept for memorabilia generation)
rooms: dict[str, Room] = {}
closed_rooms: dict[str, dict] = {}


def get_or_create_room(room_id: str, lang_a: str = "Hindi", lang_b: str = "English") -> Room:
    if room_id not in rooms:
        rooms[room_id] = Room(room_id, lang_a, lang_b)
    return rooms[room_id]


def _archive_room(room: Room):
    """Preserve room data after all participants leave for memorabilia generation."""
    closed_rooms[room.room_id] = {
        "room_id": room.room_id,
        "lang_a": room.lang_a,
        "lang_b": room.lang_b,
        "duration": room.get_formatted_time(),
        "screenshots": room.screenshots,
        "captions": room.captions,
        "scene_descriptions": room.scene_descriptions,
        "voice_snippets": room.voice_snippets,
        "start_time": room.start_time,
        "participant_names": room.participant_history or list(room.participants.keys()),
    }


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------
@app.websocket("/ws/{room_id}/{participant_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    room_id: str,
    participant_id: str,
    language: str = "English",
):
    room = get_or_create_room(room_id)
    was_reconnect = participant_id in room.participants

    # Enforce max 2 participants per room (allow reconnect by same id)
    if participant_id not in room.participants and len(room.participants) >= 2:
        await websocket.accept()
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": "Room is full (max 2 participants).",
        }))
        await websocket.close(code=1008)
        return

    await websocket.accept()
    logger.info(f"WS connected: room={room_id} participant={participant_id} lang={language}")
    participant = Participant(websocket, participant_id, language)
    room.participants[participant_id] = participant
    if participant_id not in room.participant_order:
        room.participant_order.append(participant_id)
    if participant_id not in room.participant_history:
        room.participant_history.append(participant_id)

    # Update room languages based on participants
    room._recompute_languages()

    # Send room state snapshot to the newly connected participant
    try:
        await websocket.send_text(json.dumps({
            "type": "room_state",
            **room.room_state_payload(),
        }))
    except Exception:
        pass

    # Start Gemini session if not already active
    if room.gemini_session is None and not room.use_fallback and gemini_client:
        try:
            await room.start_gemini_session()
        except Exception as e:
            error_msg = str(e)
            if "SERVICE_DISABLED" in error_msg or "not been used in project" in error_msg:
                logger.error(
                    "Generative Language API not enabled. "
                    "Visit: https://console.developers.google.com/apis/api/"
                    "generativelanguage.googleapis.com/overview"
                )
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Generative Language API not enabled in your GCP project. "
                               "Enable it in the Cloud Console, then refresh.",
                }))
            else:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": f"Translation engine unavailable: {e}",
                }))

    # Notify others
    if was_reconnect:
        await room._broadcast(
            {"type": "participant_reconnected", "participant_id": participant_id},
            exclude=participant_id,
        )
    else:
        await room._broadcast(
            {"type": "participant_joined", "participant_id": participant_id},
            exclude=participant_id,
        )

    try:
        while True:
            message = await websocket.receive()

            # Binary = raw PCM audio
            if "bytes" in message:
                if not room.should_accept_audio(participant_id):
                    continue
                if room.use_fallback:
                    room.fallback_buffer += message["bytes"]
                    # Process every ~1s of audio (16000 samples * 2 bytes = 32000)
                    if len(room.fallback_buffer) >= 32000:
                        chunk = room.fallback_buffer
                        room.fallback_buffer = b""
                        asyncio.create_task(
                            _handle_fallback_audio(room, chunk, participant_id)
                        )
                else:
                    await room.send_audio_to_gemini(message["bytes"], participant_id)

            # Text = JSON commands
            elif "text" in message:
                try:
                    data = json.loads(message["text"])
                    msg_type = data.get("type")

                    if msg_type == "camera_activated":
                        # Manual camera activation — open a 30-second vision window
                        room._activate_vision_window(participant_id, duration_s=30.0)
                        logger.info(f"Room {room_id}: Camera manually activated by {participant_id}")

                    elif msg_type == "video_frame":
                        frame_bytes = base64.b64decode(data["data"])
                        peer_id = room._peer_id(participant_id)
                        if peer_id:
                            await room._send_to_target_or_all({
                                "type": "remote_frame",
                                "data": data["data"],
                                "participant_id": participant_id,
                            }, target_id=peer_id)

                        # Process with Vision API: use fast debounce in trigger
                        # window, slower debounce (5s) for ambient camera
                        if room._vision_allowed(participant_id):
                            asyncio.create_task(room.analyze_frame_with_vision(frame_bytes))
                            asyncio.create_task(
                                room.narrate_frame_with_gemini(
                                    frame_bytes, room.last_speaker_id
                                )
                            )
                        elif room._ambient_vision_allowed(participant_id):
                            asyncio.create_task(room.analyze_frame_with_vision(frame_bytes))

                    elif msg_type == "ping":
                        await websocket.send_text(json.dumps({"type": "pong"}))

                except json.JSONDecodeError:
                    pass

    except WebSocketDisconnect:
        logger.info(f"WS disconnected: room={room_id} participant={participant_id}")
    except Exception as e:
        logger.error(f"WS error: room={room_id} participant={participant_id}: {e}")
    finally:
        room.participants.pop(participant_id, None)
        if participant_id in room.participant_order:
            room.participant_order.remove(participant_id)
        room._recompute_languages()
        await room._broadcast(
            {"type": "participant_left", "participant_id": participant_id}
        )
        # Archive + close room if empty
        if not room.participants:
            _archive_room(room)
            await room.close()
            rooms.pop(room_id, None)


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    return {
        "ready": True,
        "gemini_configured": gemini_client is not None,
        "vision_configured": vision_client is not None,
        "elevenlabs_configured": ELEVENLABS_API_KEY is not None,
    }


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.get("/room/{room_id}")
async def room_page(room_id: str):
    return FileResponse("static/participant.html")


@app.get("/api/billing")
async def billing_info():
    """Get current billing summary and cost estimates."""
    return {
        "summary": get_billing_summary(),
        "estimates": estimate_costs(),
    }


@app.get("/api/rooms")
async def list_rooms():
    return {
        room_id: {
            "participants": list(room.participants.keys()),
            "lang_a": room.lang_a,
            "lang_b": room.lang_b,
        }
        for room_id, room in rooms.items()
    }


def _get_room_data(room_id: str) -> dict | None:
    """Get room data from active or archived rooms."""
    if room_id in rooms:
        r = rooms[room_id]
        return {
            "screenshots": r.screenshots,
            "voice_snippets": r.voice_snippets,
            "scene_descriptions": r.scene_descriptions,
            "duration": r.get_formatted_time(),
            "participants": list(r.participants.keys()),
            "lang_a": r.lang_a,
            "lang_b": r.lang_b,
            "captions": r.captions,
            "active": True,
        }
    if room_id in closed_rooms:
        d = closed_rooms[room_id]
        return {
            "screenshots": d["screenshots"],
            "voice_snippets": d["voice_snippets"],
            "scene_descriptions": d["scene_descriptions"],
            "duration": d["duration"],
            "participants": d["participant_names"],
            "lang_a": d["lang_a"],
            "lang_b": d["lang_b"],
            "captions": d["captions"],
            "active": False,
        }
    return None


@app.get("/api/rooms/{room_id}/info")
async def room_info(room_id: str):
    """Room metadata for the memories page."""
    data = _get_room_data(room_id)
    if not data:
        return JSONResponse({"error": "Room not found"}, status_code=404)
    return {
        "room_id": room_id,
        "duration": data["duration"],
        "participants": data["participants"],
        "languages": [data["lang_a"], data["lang_b"]],
        "screenshots_count": len(data["screenshots"]),
        "voice_snippets_count": len(data["voice_snippets"]),
        "captions_count": len(data["captions"]),
        "active": data["active"],
    }


@app.get("/api/rooms/{room_id}/bundle")
async def room_bundle(room_id: str):
    """Full room data export for memorabilia pipelines."""
    data = _get_room_data(room_id)
    if not data:
        return JSONResponse({"error": "Room not found"}, status_code=404)
    return {
        "room_id": room_id,
        "active": data["active"],
        "duration": data["duration"],
        "participants": data["participants"],
        "languages": [data["lang_a"], data["lang_b"]],
        "screenshots": data["screenshots"],
        "captions": data["captions"],
        "scene_descriptions": data["scene_descriptions"],
        "voice_snippets": data["voice_snippets"],
    }


@app.get("/api/rooms/{room_id}/captions")
async def room_captions(room_id: str):
    """Full caption text for the memories page."""
    data = _get_room_data(room_id)
    if not data:
        return JSONResponse({"error": "Room not found"}, status_code=404)
    return {
        "room_id": room_id,
        "captions": [c.get("text", "") for c in data["captions"]],
        "voice_snippets": data["voice_snippets"],
    }


@app.post("/api/rooms/{room_id}/seed")
async def seed_room(room_id: str, body: dict = None):
    """Seed a room with pre-built data (screenshots, snippets, scenes) for demo purposes.

    POST JSON body:
    {
        "lang_a": "Chinese",
        "lang_b": "English",
        "participants": ["Nai Nai", "Billi"],
        "screenshots": [{"data": "<base64>", "description": "..."}],
        "voice_snippets": [{"text": "...", "timestamp": "00:01", "speaker": "Nai Nai"}],
        "scene_descriptions": [{"timestamp": "00:02", "description": "..."}],
        "captions": [{"text": "..."}]
    }
    """
    if body is None:
        body = {}

    room = get_or_create_room(room_id, body.get("lang_a", "Chinese"), body.get("lang_b", "English"))
    room.lang_a = body.get("lang_a", room.lang_a)
    room.lang_b = body.get("lang_b", room.lang_b)

    for ss in body.get("screenshots", []):
        room.screenshots.append({
            "data": ss.get("data", ""),
            "participant": "seed",
            "timestamp": ss.get("timestamp", room.get_formatted_time()),
            "description": ss.get("description", "Seeded frame"),
        })

    for vs in body.get("voice_snippets", []):
        room.voice_snippets.append(vs)

    for sd in body.get("scene_descriptions", []):
        room.scene_descriptions.append(sd)

    for cap in body.get("captions", []):
        room.captions.append(cap)

    # Also archive immediately so it's available on the memories page
    # even without an active WebSocket session
    _archive_room(room)

    return {
        "status": "seeded",
        "room_id": room_id,
        "screenshots": len(room.screenshots),
        "voice_snippets": len(room.voice_snippets),
        "scene_descriptions": len(room.scene_descriptions),
        "captions": len(room.captions),
    }


@app.post("/api/rooms/{room_id}/storybook", response_class=HTMLResponse)
async def create_storybook(room_id: str):
    data = _get_room_data(room_id)
    if not data:
        return HTMLResponse("<h1>Room not found</h1>", status_code=404)

    storybook_input = {
        "screenshots": data["screenshots"],
        "voice_snippets": data["voice_snippets"],
        "scene_descriptions": data["scene_descriptions"],
        "call_metadata": {
            "duration": data["duration"],
            "participants": data["participants"],
            "languages": [data["lang_a"], data["lang_b"]],
        },
    }

    num_screenshots = len(data["screenshots"])
    log_api_call("gemini_flash_image_input", num_screenshots * 0.001, f"Storybook input {room_id}")

    async with _gemini_sem:
        pages = await generate_storybook(gemini_client, storybook_input)
    if not pages:
        return HTMLResponse("<h1>Failed to generate storybook</h1>", status_code=500)
    
    # Count generated images
    num_generated = sum(1 for p in pages if p["type"] == "image")
    log_api_call("gemini_flash_image_output", num_generated, f"Storybook output {room_id}")

    html = render_storybook_html(
        pages, title=f"Our Moment Together ({data['duration']})"
    )
    return HTMLResponse(html)

@app.post("/api/rooms/{room_id}/memory-video")
async def create_memory_video(room_id: str):
    """Generate stylized images + memory video from room screenshots."""
    data = _get_room_data(room_id)
    if not data:
        return JSONResponse({"error": "Room not found"}, status_code=404)
    if not gemini_client:
        return JSONResponse({"error": "No Gemini client"}, status_code=500)

    participants = ", ".join(data["participants"]) or "two loved ones"
    num_screenshots = len(data["screenshots"])
    
    # Log stylization cost
    log_api_call("gemini_flash_image_output", num_screenshots, f"Memory video stylization {room_id}")

    result = await run_memory_video_pipeline(
        gemini_client, data["screenshots"], participants
    )
    
    # Log video generation cost if video was created
    if result["video"]:
        log_api_call("veo_video_sec", 8, f"Memory video generation {room_id}")

    response_data = {
        "stylized_count": len(result["stylized_images"]),
        "stylized_images": [],
        "video": None,
    }

    for img in result["stylized_images"]:
        img_data = img["data"]
        if isinstance(img_data, bytes):
            img_data = base64.b64encode(img_data).decode()
        response_data["stylized_images"].append({
            "data": img_data,
            "mime_type": img["mime_type"],
        })

    if result["video"]:
        vid_data = result["video"]["data"]
        if isinstance(vid_data, bytes):
            vid_data = base64.b64encode(vid_data).decode()
        response_data["video"] = {
            "data": vid_data,
            "mime_type": result["video"]["mime_type"],
        }

    return JSONResponse(response_data)


@app.post("/api/rooms/{room_id}/generate-all")
async def generate_all_memorabilia(room_id: str):
    """Launch storybook + memory video generation in parallel as a background task.

    Returns a task_id immediately. Frontend polls /generation-status/{task_id}.
    Storybook and video run concurrently via asyncio.gather — neither blocks the
    other, and both are gated by the global Gemini semaphore so they don't
    conflict on API rate limits.
    """
    data = _get_room_data(room_id)
    if not data:
        return JSONResponse({"error": "Room not found"}, status_code=404)

    task_id = f"{room_id}_{int(time.time())}"
    _generation_tasks[task_id] = {
        "status": "running",
        "room_id": room_id,
        "storybook": {"status": "pending"},
        "video": {"status": "pending"},
        "captions": {"status": "pending"},
        "started_at": time.time(),
    }

    asyncio.create_task(_run_parallel_generation(task_id, room_id, data))
    return {"task_id": task_id, "status": "started"}


@app.get("/api/rooms/{room_id}/generation-status/{task_id}")
async def generation_status(room_id: str, task_id: str):
    """Poll for generation progress. Each sub-task reports its own status."""
    task = _generation_tasks.get(task_id)
    if not task:
        return JSONResponse({"error": "Task not found"}, status_code=404)
    elapsed = time.time() - task.get("started_at", time.time())
    return {**task, "elapsed_s": round(elapsed, 1)}


async def _run_parallel_generation(task_id: str, room_id: str, data: dict):
    """Run storybook, memory video, and caption export in parallel."""
    task = _generation_tasks[task_id]

    async def gen_storybook():
        task["storybook"]["status"] = "running"
        try:
            async with _gemini_sem:
                storybook_input = {
                    "screenshots": data["screenshots"],
                    "voice_snippets": data["voice_snippets"],
                    "scene_descriptions": data["scene_descriptions"],
                    "call_metadata": {
                        "duration": data["duration"],
                        "participants": data["participants"],
                        "languages": [data["lang_a"], data["lang_b"]],
                    },
                }
                pages = await generate_storybook(gemini_client, storybook_input)
            if pages:
                html = render_storybook_html(
                    pages, title=f"Our Moment Together ({data['duration']})"
                )
                task["storybook"] = {"status": "done", "html": html}
            else:
                task["storybook"] = {"status": "empty"}
        except Exception as e:
            logger.error(f"Parallel storybook error: {e}")
            task["storybook"] = {"status": "error", "message": str(e)}

    async def gen_video():
        task["video"]["status"] = "running"
        try:
            if not gemini_client or not data["screenshots"]:
                task["video"] = {"status": "empty"}
                return
            participants = ", ".join(data["participants"]) or "two loved ones"
            result = await run_memory_video_pipeline(
                gemini_client, data["screenshots"], participants
            )
            video_data: dict = {
                "status": "done",
                "stylized_count": len(result["stylized_images"]),
                "stylized_images": [],
                "video": None,
            }
            for img in result["stylized_images"]:
                d = img["data"]
                if isinstance(d, bytes):
                    d = base64.b64encode(d).decode()
                video_data["stylized_images"].append({"data": d, "mime_type": img["mime_type"]})
            if result["video"]:
                vd = result["video"]["data"]
                if isinstance(vd, bytes):
                    vd = base64.b64encode(vd).decode()
                video_data["video"] = {"data": vd, "mime_type": result["video"]["mime_type"]}
            task["video"] = video_data
        except Exception as e:
            logger.error(f"Parallel video error: {e}")
            task["video"] = {"status": "error", "message": str(e)}

    async def gen_captions():
        task["captions"]["status"] = "running"
        try:
            task["captions"] = {
                "status": "done",
                "captions": [c.get("text", "") for c in data["captions"]],
                "voice_snippets": data["voice_snippets"],
            }
        except Exception as e:
            task["captions"] = {"status": "error", "message": str(e)}

    await asyncio.gather(
        gen_storybook(),
        gen_video(),
        gen_captions(),
        return_exceptions=True,
    )
    task["status"] = "done"
    logger.info(f"Parallel generation complete for {room_id} (task {task_id})")


@app.get("/room/{room_id}/memories", response_class=HTMLResponse)
async def memories_page(room_id: str):
    """Post-call memorabilia page — storybook + memory video."""
    return FileResponse("static/memories.html")


# ---------------------------------------------------------------------------
# ElevenLabs fallback handler
# ---------------------------------------------------------------------------
async def _handle_fallback_audio(room: Room, audio_chunk: bytes, sender_id: str):
    """Process audio through ElevenLabs STT → translate via Gemini text → ElevenLabs TTS."""
    try:
        room._mark_speaker(sender_id)
        room.last_speaker_id = sender_id
        room.last_speaker_ts = time.time()

        text = await elevenlabs_stt(audio_chunk)
        if not text or not text.strip():
            return

        # Use Gemini for text translation (cheaper than Live API)
        translated = text
        if gemini_client:
            sender_lang = "Unknown"
            for p in room.participants.values():
                if p.participant_id == sender_id:
                    sender_lang = p.language
                    break

            target_lang = room.lang_b if sender_lang == room.lang_a else room.lang_a

            try:
                translate_resp = await gemini_client.aio.models.generate_content(
                    model="gemini-2.5-flash-lite",
                    contents=f"Translate from {sender_lang} to {target_lang}. "
                             f"Return ONLY the translation:\n\n{text}",
                )
                translated = translate_resp.text.strip() if translate_resp.text else text
            except Exception as te:
                logger.warning(f"Gemini text translation failed, passing raw: {te}")
                translated = text

        await room._broadcast({
            "type": "caption",
            "text": translated,
        }, exclude=sender_id)

        tts_audio = await elevenlabs_tts(translated)
        if tts_audio:
            audio_b64 = base64.b64encode(tts_audio).decode()
            await room._broadcast({
                "type": "audio",
                "data": audio_b64,
                "mime_type": "audio/pcm;rate=16000",
            }, exclude=sender_id)

    except Exception as e:
        logger.error(f"Fallback audio handler error: {e}")


# Static files (must be last)
app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
