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
import struct
import zlib
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
    vision_opts = client_options_lib.ClientOptions(
        quota_project_id=os.getenv("GOOGLE_CLOUD_PROJECT", "974516981471")
    )
    vision_client = vision.ImageAnnotatorClient(client_options=vision_opts)
except Exception as e:
    logger.warning(f"Failed to create Vision client: {e}")
    vision_client = None

TRANSLATION_MODEL = "gemini-2.5-flash-native-audio-latest"

# ---------------------------------------------------------------------------
# Gradium fallback (STT + TTS when Gemini Live is rate-limited or recovering)
# Gradium supports: English, French, Spanish, German, Portuguese
# For Mandarin/Hindi/other: Gemini handles the translation; Gradium voices the
# English (or other supported) output side only.
# ---------------------------------------------------------------------------
GRADIUM_API_KEY = os.getenv("GRADIUM_API_KEY")

_GRADIUM_BASE = "https://us.api.gradium.ai/api/"  # key is provisioned on US servers

# Language → Gradium flagship voice (feminine / warm default per language)
_GRADIUM_VOICE_MAP: dict[str, str] = {
    "english":    "YTpq7expH9539ERJ",  # Emma (US)
    "french":     "b35yykvVppLXyw_l",  # Elise (FR)
    "spanish":    "B36pbz5_UoWn4BDl",  # Valentina (MX)
    "german":     "-uP9MuGtBqAvEyxI",  # Mia (DE)
    "portuguese": "pYcGZz9VOo4n2ynh",  # Alice (BR)
}
_GRADIUM_DEFAULT_VOICE = "YTpq7expH9539ERJ"  # Emma — English fallback

# Languages Gradium TTS can voice (map to canonical key)
_GRADIUM_SUPPORTED = {"english", "french", "spanish", "german", "portuguese"}


def _gradium_voice_for(language: str) -> str | None:
    """Return a Gradium voice_id for the given language, or None if unsupported."""
    key = language.lower().strip()
    return _GRADIUM_VOICE_MAP.get(key)


def _gradium_client():
    """Create a fresh GradiumClient pointed at the US region (where the key is provisioned)."""
    import gradium
    return gradium.client.GradiumClient(
        api_key=GRADIUM_API_KEY,
        base_url="https://us.api.gradium.ai/api/",
    )


async def gradium_stt(audio_bytes: bytes) -> str | None:
    """Transcribe audio via Gradium STT (streaming WebSocket).

    Accepts PCM 16kHz mono (our browser format) wrapped as WAV.
    Returns the full transcript or None on failure.

    Note: Gradium STT requires a paid-tier API key with WebSocket access.
    If auth fails (free tier), this returns None gracefully and the caller
    logs a warning — Gemini Live transcription remains the primary STT path.
    """
    if not GRADIUM_API_KEY:
        return None
    try:
        client = _gradium_client()
        wav = _pcm_to_wav(audio_bytes, sample_rate=16000)

        async def _audio_gen():
            chunk_size = 8192
            for i in range(0, len(wav), chunk_size):
                yield wav[i : i + chunk_size]

        stream = await client.stt_stream(
            {"model_name": "default", "input_format": "wav"},
            _audio_gen(),
        )
        parts: list[str] = []
        async for text in stream.iter_text():
            if text and text.strip():
                parts.append(text.strip())
        result = " ".join(parts).strip()
        if result:
            log_api_call("gradium_stt_sec", len(audio_bytes) / 32000.0, "Gradium STT")
        return result or None
    except Exception as e:
        err_str = str(e)
        if "Invalid or expired API key" in err_str or "1008" in err_str:
            logger.warning(
                "Gradium STT: WebSocket auth rejected — STT requires a paid Gradium plan. "
                "Gemini Live handles primary transcription; Gradium TTS still active."
            )
        else:
            logger.error(f"Gradium STT error: {type(e).__name__}: {err_str[:120]}")
    return None


async def gradium_tts(text: str, language: str = "English") -> bytes | None:
    """Generate speech via Gradium TTS POST endpoint.

    Returns PCM bytes at 16kHz (matching Gemini Live output format) or None.
    Only called when the target language is supported by Gradium.
    """
    if not GRADIUM_API_KEY:
        return None
    voice_id = _gradium_voice_for(language) or _GRADIUM_DEFAULT_VOICE
    try:
        import httpx
        async with httpx.AsyncClient(timeout=12.0) as http:
            resp = await http.post(
                f"{_GRADIUM_BASE}post/speech/tts",
                headers={
                    "x-api-key": GRADIUM_API_KEY,
                    "Content-Type": "application/json",
                },
                json={
                    "text": text,
                    "voice_id": voice_id,
                    "output_format": "pcm_16000",
                    "only_audio": True,
                },
            )
            if resp.status_code == 200:
                log_api_call("gradium_tts_char", len(text), "Gradium TTS fallback")
                return resp.content  # raw PCM 16kHz bytes
            else:
                logger.warning(f"Gradium TTS HTTP {resp.status_code}: {resp.text[:120]}")
    except Exception as e:
        logger.error(f"Gradium TTS error: {type(e).__name__}: {e}")
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

        # Mood & stylization
        self.mood: str = "sentimental"  # "sentimental" or "funny"
        self.stylized_images: list[dict] = []  # bg-stylized captures
        self._stylize_tasks: list[asyncio.Task] = []

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
        """Connect to Gemini Live API; activate Gradium fallback if it fails."""
        if not gemini_client:
            logger.error("No Gemini client — cannot start session")
            self.use_fallback = True
            return

        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Aoede")
                )
            ),
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
            if GRADIUM_API_KEY:
                logger.info(f"Room {self.room_id}: Falling back to Gradium TTS pipeline")
                self.use_fallback = True
                self.gemini_session = None
            else:
                raise

    async def _receive_loop(self):
        """Receive translated audio + transcripts from Gemini and forward.

        Auto-restarts up to 3 times on transient errors before falling back.
        """
        max_retries = 3
        retries = 0
        output_bytes_total = 0

        while True:
            if not self.gemini_session:
                return

            turn_speaker_id: Optional[str] = None
            turn_target_id: Optional[str] = None
            translated_text: Optional[str] = None
            original_text: Optional[str] = None
            saw_message = False

            try:
                # NOTE: google-genai AsyncSession.receive() yields one complete turn.
                # Keep calling it so the room continues streaming after each turn.
                async for response in self.gemini_session.receive():
                    saw_message = True
                    if not response.server_content:
                        continue

                    sc = response.server_content
                    if turn_speaker_id is None:
                        turn_speaker_id = self._speaker_for_current_turn()
                        if turn_speaker_id:
                            turn_target_id = self._peer_id(turn_speaker_id)

                    # ── Translated audio (model output) ──────────────────────
                    model_turn = sc.model_turn
                    if model_turn and model_turn.parts:
                        for part in model_turn.parts:
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
                                }, turn_target_id)

                    # Keep latest transcript chunk, then emit once per completed turn.
                    out_tx = sc.output_transcription
                    if out_tx and out_tx.text:
                        translated_text = out_tx.text
                    in_tx = sc.input_transcription
                    if in_tx and in_tx.text:
                        original_text = in_tx.text

                if not saw_message:
                    await asyncio.sleep(0.05)
                    continue

                if translated_text:
                    caption = {
                        "type": "caption",
                        "text": translated_text,
                        "speaker": turn_speaker_id or "",
                        "side": "theirs",
                    }
                    async with self._state_lock:
                        self.captions.append(caption)
                    await self._send_to_target_or_all(caption, turn_target_id)

                    text_lower = translated_text.lower()
                    if any(kw in text_lower for kw in ["love", "miss", "haha", "laugh", "beautiful", "happy"]):
                        async with self._state_lock:
                            self.voice_snippets.append({
                                "text": translated_text,
                                "timestamp": self.get_formatted_time(),
                                "speaker": "Translated Voice",
                            })
                            
                    # Start laughing narration
                    if "haha" in text_lower or "laugh" in text_lower:
                        now = time.time()
                        if (now - getattr(self, "_last_laugh_narration_ts", 0)) > 15.0:
                            self._last_laugh_narration_ts = now
                            asyncio.create_task(self._narrate_action("Wow you guys are a riot", turn_speaker_id))

                    if check_trigger(translated_text):
                        self._activate_vision_window(turn_target_id)
                        await self._broadcast({
                            "type": "trigger",
                            "trigger": "visual",
                            "text": translated_text,
                            "target_participant_id": turn_target_id,
                        })

                if original_text and turn_speaker_id and turn_speaker_id in self.participants:
                    try:
                        await self.participants[turn_speaker_id].ws.send_text(
                            json.dumps({
                                "type": "caption",
                                "text": original_text,
                                "speaker": turn_speaker_id,
                                "side": "mine",
                            })
                        )
                    except Exception:
                        pass

                retries = 0
            except asyncio.CancelledError:
                return
            except Exception as e:
                retries += 1
                logger.warning(
                    f"Room {self.room_id}: Receive loop error "
                    f"(attempt {retries}/{max_retries}): {e}"
                )
                if retries <= max_retries:
                    await asyncio.sleep(1.0)
                    try:
                        await self._reconnect_gemini_session()
                    except Exception as re:
                        logger.error(f"Room {self.room_id}: Reconnect failed: {re}")
                else:
                    logger.error(
                        f"Room {self.room_id}: Receive loop exhausted retries, switching to fallback"
                    )
                    self.use_fallback = True
                    self.gemini_session = None
                    return

    async def _reconnect_gemini_session(self):
        """Tear down and re-establish the Gemini Live session."""
        if self._session_context:
            try:
                await self._session_context.__aexit__(None, None, None)
            except Exception:
                pass
        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Aoede")
                )
            ),
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

    async def analyze_frame_with_vision(self, frame_bytes: bytes, sender_id: Optional[str] = None):
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
                        vision.Feature(type_=vision.Feature.Type.OBJECT_LOCALIZATION, max_results=5),
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
                        "mime_type": "image/jpeg",
                        "timestamp": self.get_formatted_time(),
                        "participant": self.vision_target_id or "unknown",
                        "description": caption_text or "Captured frame",
                    })

            # Emotion and Object Bounding Boxes
            vision_boxes = []

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

                # Broadcast sentiment badge for strong emotions
                if dominant_score >= 4:
                    await self._broadcast({
                        "type": "sentiment",
                        "emotion": dominant_name,
                        "score": dominant_score,
                    })

                # Build bounding box only for faces with detectable emotion (>= POSSIBLE)
                if dominant_score >= 2:
                    color = "purple"
                    label = dominant_name.capitalize()

                    if dominant_name == "happiness":
                        color = "green"
                        label = "😊 Content/Happy"
                    elif dominant_name == "anger":
                        color = "red"
                        label = "😠 Angry/Frustrated"
                    elif dominant_name in ["sadness", "surprise"]:
                        color = "purple"
                        label = "😟 Worried/Anxious"

                    vertices = [{"x": v.x, "y": v.y} for v in face.bounding_poly.vertices]

                    vision_boxes.append({
                        "vertices": vertices,
                        "color": color,
                        "label": label,
                        "type": "face",
                        "normalized": False
                    })

                    # Trigger narration for big smiles
                    if dominant_name == "happiness" and dominant_score >= 4 and is_key_moment:
                        name = sender_id.split('_')[0] if sender_id else "Someone"
                        narration = f"Wow, {name} is really happy!"
                        asyncio.create_task(self._narrate_action(narration, sender_id))

            for obj in response.localized_object_annotations:
                name_lower = obj.name.lower()
                if name_lower in ["dog", "cat", "animal", "bird", "tattoo", "hand", "finger", "person"]:
                    vertices = [{"x": v.x, "y": v.y} for v in obj.bounding_poly.normalized_vertices]
                    # Assign context-appropriate colors
                    obj_color = "green"
                    obj_label = obj.name.capitalize()
                    if name_lower in ["dog", "cat", "animal", "bird"]:
                        obj_label = f"🐾 {obj.name.capitalize()}"
                    elif name_lower == "tattoo":
                        obj_label = "🎨 Tattoo"
                    elif name_lower in ["hand", "finger"]:
                        obj_label = f"👋 {obj.name.capitalize()}"
                    elif name_lower == "person":
                        obj_label = "👤 Person"

                    vision_boxes.append({
                        "vertices": vertices,
                        "color": obj_color,
                        "label": obj_label,
                        "type": "object",
                        "normalized": True
                    })

                    if name_lower == "tattoo" or ("tattoo" in [l.description.lower() for l in response.label_annotations]):
                        if is_key_moment:
                            asyncio.create_task(self._narrate_action("Cool thing you're showing!", sender_id))

            # Send all boxes + scene labels to frontend
            if vision_boxes:
                await self._broadcast({
                    "type": "vision_boxes",
                    "boxes": vision_boxes,
                    "scene_labels": top_labels,
                })

        except Exception as e:
            logger.error(f"Room {self.room_id}: Vision API error: {e}")

    async def _narrate_action(self, text: str, speaker_id: Optional[str]):
        """Helper to narrate a specific action phrase."""
        if not GRADIUM_API_KEY:
            return
        try:
            tts_audio = await gradium_tts(text, language="English")
            if tts_audio:
                audio_b64 = base64.b64encode(tts_audio).decode()
                await self._broadcast({
                    "type": "audio",
                    "data": audio_b64,
                    "mime_type": "audio/pcm;rate=16000",
                })
                # Add to transcript for UI
                await self._broadcast({
                    "type": "narration",
                    "text": f"*{text}*",
                })
        except Exception as e:
            logger.error(f"Action narration error: {e}")

    def _narrator_fallback_text(self, prompt: str) -> str:
        """Local backup narrator response when model generation is unavailable."""
        p = (prompt or "").strip()
        if not p:
            return "Narrator is listening."
        if "summary" in p.lower():
            return "Narrator: A warm call, small details, big feelings, and a gentle goodbye."
        if "mood" in p.lower():
            return f"Narrator: Right now the room feels {self.mood}."
        return f"Narrator: {p[:140]}"

    async def narrate_storybook_prompt(self, prompt: str, requester_id: str):
        """Answer an interactive 'Narrator' prompt using recent room context."""
        prompt = (prompt or "").strip()
        if not prompt:
            return

        recent_caps = [
            c.get("text", "") for c in self.captions[-10:]
            if isinstance(c, dict) and c.get("text")
        ]
        recent_scenes = [
            s.get("description", "") for s in self.scene_descriptions[-5:]
            if isinstance(s, dict) and s.get("description")
        ]
        fallback = self._narrator_fallback_text(prompt)
        out = fallback

        if gemini_client:
            try:
                context = (
                    "You are Narrator, a warm storybook voice in a live bilingual call.\n"
                    "Reply in 1-2 short sentences, emotionally vivid but concise.\n"
                    "Avoid lists and avoid meta commentary.\n\n"
                    f"Room mood: {self.mood}\n"
                    f"Recent captions: {' | '.join(recent_caps) if recent_caps else 'None'}\n"
                    f"Recent scenes: {' | '.join(recent_scenes) if recent_scenes else 'None'}\n\n"
                    f"User prompt: {prompt}\n\n"
                    "Narrator reply:"
                )
                async with _gemini_sem:
                    resp = await gemini_client.aio.models.generate_content(
                        model="gemini-2.5-flash-lite",
                        contents=context,
                    )
                text = (resp.text or "").strip() if resp else ""
                if text:
                    out = text
            except Exception as e:
                logger.warning(f"Room {self.room_id}: Narrator generation failed: {e}")

        await self._send_to_target_or_all(
            {"type": "narrator", "text": out},
            target_id=requester_id,
        )

    def _current_exclude_id(self) -> Optional[str]:
        """Exclude the most recent speaker to avoid echo, within a short window."""
        if self.last_speaker_id and (time.time() - self.last_speaker_ts) < 10.0:
            return self.last_speaker_id
        return None

    def _current_target_id(self) -> Optional[str]:
        """Return the intended listener if we have a recent speaker and two participants."""
        speaker_id = self._speaker_for_current_turn()
        if speaker_id and len(self.participants) == 2:
            return self._peer_id(speaker_id)
        return None

    def _speaker_for_current_turn(self) -> Optional[str]:
        """Best-effort speaker identity used for caption/audio routing."""
        now = time.time()
        if self.active_speaker_id and (now - self.active_speaker_ts) < 3.0:
            return self.active_speaker_id
        if self.last_speaker_id and (now - self.last_speaker_ts) < 10.0:
            return self.last_speaker_id
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

            # Voice narration via Gradium TTS (English output only)
            if GRADIUM_API_KEY:
                tts_audio = await gradium_tts(narration_text, language="English")
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


async def _stylize_in_background(room: Room, frame_b64: str, index: int):
    """Stylize a captured photo in the background using Gemini image model.
    
    Runs with _gemini_sem and stores result in room.stylized_images.
    """
    if not gemini_client:
        return
    mood = room.mood
    if mood == "funny":
        prompt = (
            "Transform this photograph into a hilarious, exaggerated cartoon illustration. "
            "STYLE: Bold colors, big expressive eyes, over-the-top facial expressions, "
            "comic book energy, speech-bubble-ready. Think Pixar meets political cartoon. "
            "Make it genuinely funny — amplify anything awkward or silly in the scene. "
            "Add visual humor: a surprised cat in the background, exaggerated props, etc."
        )
    else:
        prompt = (
            "Transform this photograph into a warm, dreamy watercolor illustration. "
            "STYLE: Soft golden-hour lighting, gentle washes of amber and rose, "
            "slightly blurred edges like a treasured memory. Emphasize warmth, love, "
            "and human connection. The feeling of looking at a Polaroid from the best day of your life."
        )
    try:
        async with _gemini_sem:
            response = await gemini_client.aio.models.generate_content(
                model="gemini-2.5-flash-image",
                contents=[
                    {"inline_data": {"mime_type": "image/jpeg", "data": frame_b64}},
                    {"text": prompt},
                ],
                config=types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"]),
            )
        if response.candidates and response.candidates[0].content:
            for part in response.candidates[0].content.parts:
                if hasattr(part, "inline_data") and part.inline_data:
                    img_data = part.inline_data.data
                    if isinstance(img_data, str):
                        img_data = base64.b64decode(img_data)
                    async with room._state_lock:
                        room.stylized_images.append({
                            "data": img_data,
                            "mime_type": part.inline_data.mime_type or "image/png",
                            "index": index,
                            "mood": mood,
                        })
                    logger.info(f"Room {room.room_id}: photo #{index} stylized ({mood})")
                    # Notify all participants
                    await room._broadcast({
                        "type": "photo_stylized",
                        "index": index,
                        "mood": mood,
                    })
                    return
        logger.warning(f"Room {room.room_id}: stylization #{index} returned no image (rate limited?)")
    except Exception as e:
        logger.error(f"Room {room.room_id}: stylization #{index} error: {e}")


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
        "stylized_images": room.stylized_images,
        "mood": room.mood,
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
                            asyncio.create_task(room.analyze_frame_with_vision(frame_bytes, participant_id))
                            asyncio.create_task(
                                room.narrate_frame_with_gemini(
                                    frame_bytes, room.last_speaker_id
                                )
                            )
                        elif room._ambient_vision_allowed(participant_id):
                            asyncio.create_task(room.analyze_frame_with_vision(frame_bytes, participant_id))

                    elif msg_type == "set_mood":
                        room.mood = data.get("mood", "sentimental")
                        logger.info(f"Room {room_id}: mood set to '{room.mood}'")
                        await room._broadcast({"type": "mood_set", "mood": room.mood})

                    elif msg_type == "photo_capture":
                        frame_b64 = data.get("data", "")
                        if frame_b64:
                            ts = room.get_formatted_time()
                            screenshot = {
                                "data": frame_b64,
                                "mime_type": "image/jpeg",
                                "participant": participant_id,
                                "timestamp": ts,
                                "description": "User-captured moment",
                            }
                            async with room._state_lock:
                                room.screenshots.append(screenshot)
                            idx = len(room.screenshots)
                            logger.info(f"Room {room_id}: photo #{idx} captured by {participant_id}")
                            # Notify frontend the capture was saved
                            await websocket.send_text(json.dumps({
                                "type": "photo_saved",
                                "index": idx,
                                "timestamp": ts,
                            }))
                            # Kick off background stylization
                            task = asyncio.create_task(
                                _stylize_in_background(room, frame_b64, idx)
                            )
                            room._stylize_tasks.append(task)

                    elif msg_type == "narrator_prompt":
                        prompt = (data.get("text") or "").strip()
                        if prompt:
                            asyncio.create_task(
                                room.narrate_storybook_prompt(prompt, participant_id)
                            )

                    elif msg_type == "ping":
                        await websocket.send_text(json.dumps({"type": "pong"}))

                except json.JSONDecodeError:
                    pass
                except Exception as e:
                    logger.warning(
                        f"Room {room_id}: Ignoring malformed client message from "
                        f"{participant_id}: {e}"
                    )

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
        "gradium_configured": GRADIUM_API_KEY is not None,
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
            "stylized_images": r.stylized_images,
            "mood": r.mood,
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
            "stylized_images": d.get("stylized_images", []),
            "mood": d.get("mood", "sentimental"),
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


@app.post("/api/rooms/{room_id}/mood")
async def set_room_mood(room_id: str, body: dict = None):
    """Set the mood for a room (sentimental or funny)."""
    if body is None:
        body = {}
    mood = body.get("mood", "sentimental")
    if mood not in ("sentimental", "funny"):
        return JSONResponse({"error": "mood must be 'sentimental' or 'funny'"}, status_code=400)
    if room_id in rooms:
        rooms[room_id].mood = mood
    if room_id in closed_rooms:
        closed_rooms[room_id]["mood"] = mood
    return {"room_id": room_id, "mood": mood}


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
            "mime_type": ss.get("mime_type", "image/jpeg"),
            "participant": "seed",
            "timestamp": ss.get("timestamp", room.get_formatted_time()),
            "description": ss.get("description", "Seeded frame"),
        })

    for simg in body.get("stylized_images", []):
        room.stylized_images.append({
            "data": simg.get("data", ""),
            "mime_type": simg.get("mime_type", "image/png"),
            "index": simg.get("index", len(room.stylized_images) + 1),
            "mood": simg.get("mood", room.mood),
        })

    for vs in body.get("voice_snippets", []):
        room.voice_snippets.append(vs)

    for sd in body.get("scene_descriptions", []):
        room.scene_descriptions.append(sd)

    for cap in body.get("captions", []):
        room.captions.append(cap)

    if "mood" in body:
        room.mood = body["mood"]

    if "participant_names" in body:
        room.participant_history = body["participant_names"]
    elif "participants" in body and isinstance(body["participants"], list):
        room.participant_history = [str(p) for p in body["participants"]]

    # Also archive immediately so it's available on the memories page
    _archive_room(room)

    return {
        "status": "seeded",
        "room_id": room_id,
        "screenshots": len(room.screenshots),
        "voice_snippets": len(room.voice_snippets),
        "scene_descriptions": len(room.scene_descriptions),
        "captions": len(room.captions),
        "mood": room.mood,
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
    if gemini_client:
        log_api_call("gemini_flash_image_input", num_screenshots * 0.001, f"Storybook input {room_id}")
        async with _gemini_sem:
            pages = await generate_storybook(gemini_client, storybook_input)
    else:
        pages = await generate_storybook(None, storybook_input)

    if not pages:
        return HTMLResponse("<h1>Failed to generate storybook</h1>", status_code=500)
    
    # Count generated images
    num_generated = sum(1 for p in pages if p["type"] == "image")
    if gemini_client and num_generated:
        log_api_call("gemini_flash_image_output", num_generated, f"Storybook output {room_id}")

    html = render_storybook_html(
        pages,
        title=f"Our Moment Together ({data['duration']})",
        stylized_images=data.get("stylized_images", []),
        mood=data.get("mood", "sentimental"),
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

    participant_names = [str(p).split("_")[0] for p in data["participants"]]
    participants = " and ".join(participant_names[:2]) or "two loved ones"
    mood = data.get("mood", "sentimental")

    # Use pre-stylized images if available; otherwise stylize first 2 screenshots
    pre_stylized = data.get("stylized_images", [])
    screenshots_for_video = data["screenshots"][:2] if data["screenshots"] else []
    num_screenshots = len(screenshots_for_video)
    
    log_api_call("gemini_flash_image_output", num_screenshots, f"Memory video stylization {room_id}")

    result = await run_memory_video_pipeline(
        gemini_client,
        screenshots_for_video,
        participants,
        voice_snippets=data["voice_snippets"],
        scene_descriptions=data["scene_descriptions"],
        mood=mood,
        pre_stylized=pre_stylized,
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
                    pages,
                    title=f"Our Moment Together ({data['duration']})",
                    stylized_images=data.get("stylized_images", []),
                    mood=data.get("mood", "sentimental"),
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
            participant_names = [str(p).split("_")[0] for p in data["participants"]]
            participants = " and ".join(participant_names[:2]) or "two loved ones"
            mood = data.get("mood", "sentimental")
            result = await run_memory_video_pipeline(
                gemini_client,
                data["screenshots"][:2],
                participants,
                voice_snippets=data["voice_snippets"],
                scene_descriptions=data["scene_descriptions"],
                mood=mood,
                pre_stylized=data.get("stylized_images", []),
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
# Gradium fallback handler
# Pipeline: Gradium STT → Gemini text translate → Gradium TTS (if supported)
# ---------------------------------------------------------------------------
async def _handle_fallback_audio(room: Room, audio_chunk: bytes, sender_id: str):
    """Process audio through Gradium STT → Gemini text translation → Gradium TTS.

    Gradium TTS is used only when the target language is supported (en/fr/es/de/pt).
    For Mandarin, Hindi, and other languages Gemini doesn't produce an audio output
    here — the transcript caption is still sent so the UI stays updated.
    """
    try:
        room._mark_speaker(sender_id)
        room.last_speaker_id = sender_id
        room.last_speaker_ts = time.time()

        text = await gradium_stt(audio_chunk)
        if not text or not text.strip():
            return

        # Determine sender / target languages
        sender_lang = "Unknown"
        for p in room.participants.values():
            if p.participant_id == sender_id:
                sender_lang = p.language
                break
        target_lang = room.lang_b if sender_lang == room.lang_a else room.lang_a

        # Translate via Gemini text API (far cheaper than Live API per call)
        translated = text
        if gemini_client:
            try:
                translate_resp = await gemini_client.aio.models.generate_content(
                    model="gemini-2.5-flash-lite",
                    contents=(
                        f"Translate the following from {sender_lang} to {target_lang}. "
                        f"Return ONLY the translated text, nothing else:\n\n{text}"
                    ),
                )
                translated = translate_resp.text.strip() if translate_resp.text else text
            except Exception as te:
                logger.warning(f"Gemini text translation failed in fallback, using original: {te}")

        target_id = room._peer_id(sender_id)

        # Send translated caption to listener
        translated_caption = {
            "type": "caption",
            "text": translated,
            "speaker": sender_id,
            "side": "theirs",
        }
        async with room._state_lock:
            room.captions.append(translated_caption)
        await room._send_to_target_or_all(translated_caption, target_id)

        # Send original transcript caption back to speaker
        if sender_id in room.participants:
            try:
                await room.participants[sender_id].ws.send_text(
                    json.dumps({
                        "type": "caption",
                        "text": text,
                        "speaker": sender_id,
                        "side": "mine",
                    })
                )
            except Exception:
                pass

        # Gradium TTS — only for supported output languages
        if target_lang.lower() in _GRADIUM_SUPPORTED:
            tts_audio = await gradium_tts(translated, language=target_lang)
            if tts_audio:
                audio_b64 = base64.b64encode(tts_audio).decode()
                await room._send_to_target_or_all({
                    "type": "audio",
                    "data": audio_b64,
                    "mime_type": "audio/pcm;rate=16000",
                }, target_id)
                logger.info(f"Gradium TTS delivered {len(tts_audio)} PCM bytes → {target_lang}")
        else:
            logger.info(
                f"Gradium TTS skipped (target lang '{target_lang}' not supported); "
                f"caption sent as text-only fallback."
            )

    except Exception as e:
        logger.error(f"Gradium fallback handler error: {e}")


# ---------------------------------------------------------------------------
# Demo route — seeds the grandma/granddaughter story and redirects to memories
# ---------------------------------------------------------------------------
def _png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    return (
        struct.pack("!I", len(payload))
        + chunk_type
        + payload
        + struct.pack("!I", zlib.crc32(chunk_type + payload) & 0xFFFFFFFF)
    )


def _make_gradient_png_b64(
    width: int,
    height: int,
    top_rgb: tuple[int, int, int],
    bottom_rgb: tuple[int, int, int],
) -> str:
    """Generate a simple RGB gradient PNG and return base64 payload."""
    rows = bytearray()
    den = max(1, height - 1)
    for y in range(height):
        t = y / den
        r = int(top_rgb[0] * (1 - t) + bottom_rgb[0] * t)
        g = int(top_rgb[1] * (1 - t) + bottom_rgb[1] * t)
        b = int(top_rgb[2] * (1 - t) + bottom_rgb[2] * t)
        rows.append(0)  # filter type: None
        rows.extend(bytes((r, g, b)) * width)

    ihdr = struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0)
    idat = zlib.compress(bytes(rows), level=9)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", idat)
        + _png_chunk(b"IEND", b"")
    )
    return base64.b64encode(png).decode("ascii")


_DEMO_VOICE_SNIPPETS = [
    {"text": "我想你了，小梅。每天都想你。", "translation": "I miss you so much, little Mei. I think of you every single day.", "timestamp": "00:01", "speaker": "Nǎi Nai", "emotion": "longing"},
    {"text": "吃饭了吗？你在那边一定要好好吃饭。", "translation": "Have you eaten? You must take care of yourself over there.", "timestamp": "00:03", "speaker": "Nǎi Nai", "emotion": "warmth"},
    {"text": "Nai Nai, I'm eating your dumplings right now. I made them from the recipe you wrote me.", "translation": "奶奶，我正在吃你包的饺子。我用你写给我的食谱做的。", "timestamp": "00:05", "speaker": "Mei", "emotion": "joy"},
    {"text": "真的？你包的一定很好吃。你奶奶我教出来的！", "translation": "Really? They must taste wonderful. After all, I taught you!", "timestamp": "00:06", "speaker": "Nǎi Nai", "emotion": "pride"},
    {"text": "我给你看看我家门口的花，今年开得特别好。", "translation": "Let me show you the flowers by my door — they're especially beautiful this year.", "timestamp": "00:09", "speaker": "Nǎi Nai", "emotion": "joy"},
    {"text": "Oh, Nai Nai, they're gorgeous. Are those from Grandpa's garden?", "translation": "哇，奶奶，真漂亮。那是爷爷花园里的那些花吗？", "timestamp": "00:10", "speaker": "Mei", "emotion": "nostalgia"},
    {"text": "对，就是他种的那些。他要是看到你现在这么出息，一定很骄傲。", "translation": "Yes, the ones he planted. If he could see how well you've done, he'd be so proud.", "timestamp": "00:11", "speaker": "Nǎi Nai", "emotion": "bittersweet"},
    {"text": "I love you, Nai Nai. I'll come home soon.", "translation": "我爱你，奶奶。我很快就回家。", "timestamp": "00:14", "speaker": "Mei", "emotion": "love"},
    {"text": "我爱你，小梅。路上小心。奶奶在这里等你。", "translation": "I love you too, little Mei. Travel safe. Grandma will be right here waiting.", "timestamp": "00:15", "speaker": "Nǎi Nai", "emotion": "love"},
]

_DEMO_SCENE_DESCRIPTIONS = [
    {"timestamp": "00:02", "description": "An elderly woman's face glows warmly in a Beijing apartment. Faded red paper cuttings hang on the window behind her. A teapot steams on the table. Afternoon light filters through sheer curtains.", "labels": ["indoor", "warm", "Beijing", "traditional"]},
    {"timestamp": "00:05", "description": "A young woman holds up a plate of freshly made dumplings toward the camera in a New York studio apartment. She's smiling, flour still on her hands. A handwritten recipe card is visible on the counter.", "labels": ["food", "cooking", "joy", "New York"]},
    {"timestamp": "00:09", "description": "The grandmother points her phone at red and pink roses blooming beside a wooden gate opening onto a narrow Beijing hutong alley. Morning light catches the petals.", "labels": ["outdoor", "flowers", "garden", "Beijing"]},
    {"timestamp": "00:12", "description": "The grandmother holds up a framed photograph — a younger version of herself standing with a man in a garden. She points to the flowers in the background, which match the roses she just showed.", "labels": ["portrait", "memory", "family", "nostalgia"]},
    {"timestamp": "00:15", "description": "Both screens in split view: Nǎi Nai pressing her palm to the camera glass in Beijing, and Mei pressing hers in New York. Two palms, one connection, six thousand miles apart.", "labels": ["gesture", "connection", "emotional", "farewell"]},
]

_DEMO_CAPTIONS = [
    {"text": "I miss you so much, little Mei. I think of you every single day.", "side": "theirs"},
    {"text": "Have you eaten? You must take care of yourself over there.", "side": "theirs"},
    {"text": "Nai Nai, I'm eating your dumplings right now!", "side": "mine"},
    {"text": "Really? They must taste wonderful. After all, I taught you!", "side": "theirs"},
    {"text": "Let me show you the flowers by my door — they're especially beautiful this year.", "side": "theirs"},
    {"text": "Oh Nai Nai, they're gorgeous. Are those from Grandpa's garden?", "side": "mine"},
    {"text": "Yes. If he could see how well you've done, he'd be so proud.", "side": "theirs"},
    {"text": "I love you, Nai Nai. I'll come home soon.", "side": "mine"},
    {"text": "I love you too, little Mei. Grandma will be right here waiting.", "side": "theirs"},
]

_DEMO_SCREENSHOTS = [
    {
        "data": _make_gradient_png_b64(640, 360, (35, 24, 28), (152, 108, 84)),
        "mime_type": "image/png",
        "timestamp": "00:02",
        "description": _DEMO_SCENE_DESCRIPTIONS[0]["description"],
    },
    {
        "data": _make_gradient_png_b64(640, 360, (28, 40, 58), (194, 146, 112)),
        "mime_type": "image/png",
        "timestamp": "00:08",
        "description": _DEMO_SCENE_DESCRIPTIONS[2]["description"],
    },
    {
        "data": _make_gradient_png_b64(640, 360, (22, 20, 30), (132, 86, 120)),
        "mime_type": "image/png",
        "timestamp": "00:14",
        "description": _DEMO_SCENE_DESCRIPTIONS[4]["description"],
    },
]


def _seed_grandma_demo_room(room: Room) -> bool:
    """Populate demo room artifacts for memory/storybook/video demo."""
    changed = False
    room.lang_a = "Mandarin Chinese"
    room.lang_b = "English"
    room.mood = "sentimental"

    if not room.voice_snippets:
        room.voice_snippets.extend(_DEMO_VOICE_SNIPPETS)
        changed = True
    if not room.scene_descriptions:
        room.scene_descriptions.extend(_DEMO_SCENE_DESCRIPTIONS)
        changed = True
    if not room.captions:
        room.captions.extend(_DEMO_CAPTIONS)
        changed = True
    if not room.screenshots:
        room.screenshots.extend([{**ss, "participant": "seed"} for ss in _DEMO_SCREENSHOTS])
        changed = True
    if not room.stylized_images:
        room.stylized_images.extend([
            {
                "data": ss["data"],
                "mime_type": ss.get("mime_type", "image/png"),
                "index": i + 1,
                "mood": "sentimental",
            }
            for i, ss in enumerate(_DEMO_SCREENSHOTS[:2])
        ])
        changed = True
    if not room.participant_history:
        room.participant_history = ["Nǎi Nai", "Mei"]
        changed = True

    if changed:
        _archive_room(room)
    return changed


@app.get("/demo/live")
async def demo_live_redirect():
    """Separate URL for live translation demo."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/?room=live-demo", status_code=302)


@app.get("/demo")
async def demo_redirect():
    """Legacy demo URL: redirects to memory demo."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/demo/memory", status_code=302)


@app.get("/demo/memory")
async def demo_memory_redirect():
    """Separate URL for memory/storybook/video demo."""
    from fastapi.responses import RedirectResponse
    room_id = "grandma-demo"
    room = get_or_create_room(room_id, "Mandarin Chinese", "English")
    changed = _seed_grandma_demo_room(room)
    if changed:
        logger.info(
            f"Demo room '{room_id}' seeded with "
            f"{len(room.voice_snippets)} voice moments and {len(room.screenshots)} screenshots"
        )

    return RedirectResponse(url=f"/room/{room_id}/memories", status_code=302)


@app.post("/api/demo/reseed")
async def demo_reseed():
    """Re-seed the demo room (clears existing data first)."""
    room_id = "grandma-demo"
    if room_id in rooms:
        del rooms[room_id]
    if room_id in closed_rooms:
        del closed_rooms[room_id]

    room = get_or_create_room(room_id, "Mandarin Chinese", "English")
    _seed_grandma_demo_room(room)

    return {
        "status": "reseeded",
        "room_id": room_id,
        "memories_url": f"/room/{room_id}/memories",
        "live_url": "/demo/live",
        "voice_snippets": len(room.voice_snippets),
        "scenes": len(room.scene_descriptions),
        "screenshots": len(room.screenshots),
    }


# Static files (must be last)
app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
