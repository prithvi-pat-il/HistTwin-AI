import sys
import io
import os
import re
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

os.environ["PYTHONIOENCODING"] = "utf-8"
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Set Hugging Face token if present in environment
if os.getenv("HF_TOKEN"):
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import tempfile
import time
import subprocess
import asyncio
import edge_tts
from predict_place import predict_image, get_place_details, generate_video_prompt, generate_sora_video, general_chat
from avatar import generate_virtual_human_video

app = FastAPI()

# Enable CORS (useful for development, though we are serving static files from same origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (Moved to end)
os.makedirs("static", exist_ok=True)

# ElevenLabs API Key & Male/Boy Voice Configuration from environment
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
USER_SHARED_VOICE_ID = os.getenv("ELEVENLABS_USER_SHARED_VOICE_ID", "DJDkcaY4POaxra3iaZ5b")
BOY_VOICE_ID = os.getenv("ELEVENLABS_BOY_VOICE_ID", "TX3LPaxmHKxFdv7VOQHJ")  # Liam (Young Boy Voice)

ELEVENLABS_VOICES = {
    "English": USER_SHARED_VOICE_ID,
    "Hindi": USER_SHARED_VOICE_ID,
    "Marathi": USER_SHARED_VOICE_ID
}

# Edge-TTS / Hugging Face Voice Mapping (Male & Neural Voices)
VOICE_MAP = {
    "English": "en-US-GuyNeural",
    "Hindi": "hi-IN-MadhurNeural",
    "Marathi": "mr-IN-AarohiNeural"
}

async def synthesize_elevenlabs(text: str, language: str, output_path: str) -> bool:
    """
    Speech Synthesis using ElevenLabs Turbo v2.5 model.
    Attempts user shared voice (DJDkcaY4POaxra3iaZ5b) and falls back to Liam boy voice (TX3LPaxmHKxFdv7VOQHJ).
    """
    voice_ids_to_try = [USER_SHARED_VOICE_ID, BOY_VOICE_ID]
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }
    narration_text = text[:400] if len(text) > 400 else text

    try:
        import httpx
        async with httpx.AsyncClient(timeout=10.0) as client:
            for vid in voice_ids_to_try:
                url = f"https://api.elevenlabs.io/v1/text-to-speech/{vid}"
                payload = {
                    "text": narration_text,
                    "model_id": "eleven_turbo_v2_5",
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.75
                    }
                }
                res = await client.post(url, json=payload, headers=headers)
                if res.status_code == 200 and len(res.content) > 0:
                    with open(output_path, "wb") as f:
                        f.write(res.content)
                    voice_label = "Shared Voice" if vid == USER_SHARED_VOICE_ID else "Liam (Boy Voice)"
                    print(f"[HistTwin AI] ElevenLabs Turbo TTS ({voice_label}) successful for {language} in < 1s!")
                    return True
                else:
                    print(f"[HistTwin AI] ElevenLabs voice {vid} status {res.status_code}. Trying next...")

            return False
    except Exception as e:
        print(f"[HistTwin AI] ElevenLabs exception: {e}. Switching to fast fallback...")
        return False


async def synthesize_huggingface(text: str, language: str, output_path: str) -> bool:
    """
    Speech Synthesis using Hugging Face / Edge Neural TTS Engine.
    Provides crisp, high-speed neural audio for English, Hindi, and Marathi.
    """
    voice = VOICE_MAP.get(language, "en-US-GuyNeural")
    try:
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_path)
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(f"[HistTwin AI] Hugging Face / Edge TTS synthesis successful for {language} ({voice})!")
            return True
    except Exception as e:
        print(f"[HistTwin AI] Hugging Face / Edge-TTS failed: {e}. Trying OmniVoice / gTTS fallback...")
        try:
            from gtts import gTTS
            lang_code = "en" if language == "English" else ("hi" if language == "Hindi" else "mr")
            gTTS(text, lang=lang_code).save(output_path)
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                return True
        except Exception as ge:
            print(f"Fallback failed: {ge}")
            return False
    return False


async def generate_speech_and_motion(text: str, language: str, tts_engine: str = "elevenlabs"):
    """
    Fast speech synthesis supporting selectable TTS Engines:
    - 'elevenlabs': ElevenLabs AI Voice (Boy / Shared Voice)
    - 'huggingface': Hugging Face / Edge Neural Voice Engine
    Returns: dict containing (audio_url, video_url, motion_url, viseme_url, tts_engine)
    """
    timestamp = int(time.time() * 1000)
    mp3_filename = f"speech_{timestamp}.mp3"
    mp3_path = os.path.join("static", mp3_filename)
    
    # Cleanup old generated files in static folder
    try:
        current_time = time.time()
        for f in os.listdir("static"):
            if f.startswith(("speech_", "motion_")) and f.endswith((".mp3", ".wav", ".json")):
                fp = os.path.join("static", f)
                if os.path.isfile(fp) and (current_time - os.path.getmtime(fp)) > 600:
                    os.remove(fp)
    except Exception as e:
        print("Cleanup error:", e)
    
    # Clean text for TTS (Optimized for instant storytelling audio synthesis)
    clean_text = re.sub(r'[*#_`]', '', text)
    clean_text = clean_text.replace('\n', ' ').strip()
    if len(clean_text) > 450:
        cutoff = clean_text[:450].rfind('.')
        if cutoff > 180:
            clean_text = clean_text[:cutoff+1]
        else:
            clean_text = clean_text[:450] + '...'
    
    audio_path = None
    engine_used = tts_engine.lower()
    
    if engine_used == "huggingface" or engine_used == "hf":
        # 1. Hugging Face / Edge Neural TTS
        success = await synthesize_huggingface(clean_text, language, mp3_path)
        if success and os.path.exists(mp3_path):
            audio_path = mp3_path
        else:
            # Fallback to ElevenLabs
            el_success = await synthesize_elevenlabs(clean_text, language, mp3_path)
            if el_success and os.path.exists(mp3_path):
                audio_path = mp3_path
                engine_used = "elevenlabs"
    else:
        # 2. ElevenLabs API (Default / 1st Priority)
        el_success = await synthesize_elevenlabs(clean_text, language, mp3_path)
        if el_success and os.path.exists(mp3_path) and os.path.getsize(mp3_path) > 0:
            audio_path = mp3_path
            engine_used = "elevenlabs"
        else:
            # Fallback to Hugging Face / Edge-TTS
            hf_success = await synthesize_huggingface(clean_text, language, mp3_path)
            if hf_success and os.path.exists(mp3_path):
                audio_path = mp3_path
                engine_used = "huggingface"

    if audio_path and os.path.exists(audio_path):
        audio_url = f"/{os.path.basename(audio_path)}"
        
        # Lightweight instant motion generation
        motion_url = None
        try:
            from avatar.motion_generator import generate_motion
            motion_json = generate_motion(audio_path)
            if motion_json and os.path.exists(motion_json):
                motion_url = f"/{os.path.basename(motion_json)}"
        except Exception as me:
            print("Lightweight motion notice:", me)

        return {
            "audio_url": audio_url,
            "video_url": None,
            "motion_url": motion_url,
            "motion_data": motion_url,
            "viseme_url": None,
            "tts_engine": engine_used
        }

    return {"audio_url": None, "video_url": None, "motion_url": None, "viseme_url": None, "tts_engine": engine_used}


def split_into_lines(text: str) -> list[str]:
    if not text:
        return []
    lines = [line.strip() for line in re.split(r'(?<=[.!?\n])\s+', text) if line.strip()]
    return lines if lines else [text]

@app.post("/chat")
async def chat(
    file: UploadFile = File(None),
    text: str = Form(None),
    language: str = Form("English"),
    context_place: str = Form(None),
    tts_engine: str = Form("elevenlabs")
):
    """
    HistTwin AI Unified Chat Endpoint.
    Handles image recognition, historical story generation, and selectable ElevenLabs / Hugging Face TTS.
    """
    
    # CASE 1 & 2: Image is provided
    if file:
        suffix = os.path.splitext(file.filename)[1] if file.filename else ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = tmp_file.name

        try:
            prediction, confidence = predict_image(tmp_path)
            
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

            if not prediction:
                return JSONResponse(status_code=400, content={"error": "Could not process image."})
            
            if prediction == "Not a recognizable place":
                 msg = "I could not recognize this place. Please try another image."
                 return {
                    "type": "prediction",
                    "prediction": prediction,
                    "confidence": confidence,
                    "text": msg,
                    "response": msg,
                    "text_lines": [msg]
                 }

            # Generate Response & Speech
            if text:
                answer = get_place_details(prediction, language, user_question=text)
                avatar_res = await generate_speech_and_motion(answer, language, tts_engine=tts_engine)
                return {
                    "type": "prediction",
                    "prediction": prediction,
                    "confidence": confidence,
                    "text": answer,
                    "response": answer,
                    "text_lines": split_into_lines(answer),
                    "audio_url": avatar_res.get("audio_url"),
                    "video_url": avatar_res.get("video_url"),
                    "motion_url": avatar_res.get("motion_url"),
                    "motion_data": avatar_res.get("motion_url"),
                    "viseme_url": avatar_res.get("viseme_url"),
                    "tts_engine": avatar_res.get("tts_engine", tts_engine)
                }
            else:
                history = get_place_details(prediction, language)
                avatar_res = await generate_speech_and_motion(history, language, tts_engine=tts_engine)
                
                return {
                    "type": "prediction",
                    "prediction": prediction,
                    "confidence": confidence,
                    "text": history,
                    "response": history,
                    "text_lines": split_into_lines(history),
                    "video_url": None,
                    "audio_url": avatar_res.get("audio_url"),
                    "motion_url": avatar_res.get("motion_url"),
                    "motion_data": avatar_res.get("motion_url"),
                    "viseme_url": None,
                    "tts_engine": avatar_res.get("tts_engine", tts_engine)
                }

        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

    # CASE 3: Text only (Context required or general chat)
    elif text:
        if not context_place:
             answer = general_chat(text, language)
        else:
             answer = get_place_details(context_place, language, user_question=text)
             
        avatar_res = await generate_speech_and_motion(answer, language, tts_engine=tts_engine)
        return {
            "type": "text",
            "prediction": context_place,
            "text": answer,
            "response": answer,
            "text_lines": split_into_lines(answer),
            "audio_url": avatar_res.get("audio_url"),
            "video_url": avatar_res.get("video_url"),
            "motion_url": avatar_res.get("motion_url"),
            "motion_data": avatar_res.get("motion_url"),
            "viseme_url": avatar_res.get("viseme_url"),
            "tts_engine": avatar_res.get("tts_engine", tts_engine)
        }

    else:
        return JSONResponse(status_code=400, content={"error": "No input provided."})

app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
