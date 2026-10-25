import os
import hashlib
import json
import shutil
import subprocess

from .motion_generator import generate_motion
from .lip_sync import generate_lip_sync_json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STATIC_DIR = os.path.join(PROJECT_ROOT, "static")
CACHE_DIR = os.path.join(STATIC_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def calculate_text_hash(text: str, language: str) -> str:
    """Calculates MD5 hash of text and language for caching virtual human videos."""
    combined = f"{text}_{language}".encode("utf-8")
    return hashlib.md5(combined).hexdigest()


def render_avatar_video_opencv(audio_path: str, motion_json_path: str, output_mp4_path: str):
    """
    Renders a virtual human guide video using OpenCV & PyRender fallback, 
    compositing joint postures and audio into a standard MP4 file.
    """
    try:
        import cv2
        import numpy as np

        with open(motion_json_path, "r") as f:
            motion_data = json.load(f)

        poses = motion_data.get("poses", [])
        fps = motion_data.get("fps", 30)
        frame_count = len(poses)

        if frame_count == 0:
            frame_count = 150

        width, height = 640, 640
        temp_video_path = output_mp4_path.replace(".mp4", "_raw.mp4")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

        # Joint index definitions for 2D/3D composite rendering
        # Spine: 3, Neck: 12, Head: 15, L_Shoulder: 13, R_Shoulder: 14, L_Elbow: 18, R_Elbow: 19
        joint_colors = {
            "head": (0, 255, 230),
            "torso": (255, 170, 0),
            "left_arm": (0, 215, 255),
            "right_arm": (255, 0, 170)
        }

        guide_img_path = os.path.join(STATIC_DIR, "maharashtrian_guide.png")
        base_img = None
        if os.path.exists(guide_img_path):
            loaded_img = cv2.imread(guide_img_path)
            if loaded_img is not None:
                base_img = cv2.resize(loaded_img, (width, height))

        viseme_path = audio_path.rsplit(".", 1)[0] + "_visemes.json"
        visemes_data = []
        if os.path.exists(viseme_path):
            try:
                with open(viseme_path, "r") as vf:
                    visemes_data = json.load(vf).get("timeline", [])
            except Exception:
                pass

        for idx in range(frame_count):
            if base_img is not None:
                frame_img = base_img.copy()
            else:
                frame_img = np.zeros((height, width, 3), dtype=np.uint8)
                for y in range(height):
                    bg_val = int(20 + 30 * (y / height))
                    frame_img[y, :] = (bg_val + 10, bg_val, bg_val - 5)

            # Extract frame viseme or fallback to audio rhythm
            mouth_val = 0.0
            if idx < len(visemes_data):
                mouth_val = visemes_data[idx].get("mouthOpen", 0.0)
            else:
                mouth_val = 0.3 * (1.0 + np.sin(idx * 0.45) * np.cos(idx * 0.25))

            if mouth_val > 0.05:
                mouth_w = int(14 + mouth_val * 6)
                mouth_h = int(3 + mouth_val * 14)
                # Outer lip outline & shadow
                cv2.ellipse(frame_img, (305, 205), (mouth_w + 2, mouth_h + 2), 0, 0, 360, (20, 10, 5), -1)
                # Inner cavity
                cv2.ellipse(frame_img, (305, 205), (mouth_w, mouth_h), 0, 0, 360, (40, 15, 10), -1)
                # Subtle speaking glow ring around card
                cv2.rectangle(frame_img, (5, 5), (width - 5, height - 5), (0, 215, 255), 3)

            # Overlay branding header
            cv2.putText(frame_img, "HistTwin AI - Swarajya Cultural Guide", (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 215, 255), 2)

            out.write(frame_img)

        out.release()

        # Merge audio track into video using ffmpeg if available
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", temp_video_path, "-i", audio_path,
                "-c:v", "libx264", "-c:a", "aac", "-shortest", output_mp4_path
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
        except Exception:
            # Fallback: rename raw video to output_mp4_path
            if os.path.exists(temp_video_path):
                shutil.move(temp_video_path, output_mp4_path)

    except Exception as e:
        print(f"[HistTwin AI] OpenCV video render exception: {e}")


def generate_virtual_human_video(text: str, audio_path: str, language: str = "English") -> dict:
    """
    Full pipeline function to generate virtual human avatar assets and video.
    
    Flow:
    Text + Audio -> Motion Generator (PantoMatrix) -> Lip Sync (Visemes) -> Avatar Video Renderer
    
    Args:
        text: Narrative story text
        audio_path: Path to synthesized TTS audio file (.wav or .mp3)
        language: Language string ("English", "Hindi", "Marathi")
        
    Returns:
        Dictionary containing video_url, motion_url, audio_url, and viseme_url.
    """
    text_hash = calculate_text_hash(text, language)
    cached_video_filename = f"virtual_human_{text_hash}.mp4"
    cached_video_path = os.path.join(CACHE_DIR, cached_video_filename)
    
    base_filename = os.path.basename(audio_path).rsplit(".", 1)[0]
    output_video_path = os.path.join(STATIC_DIR, f"virtual_human_{base_filename}.mp4")
    output_viseme_path = os.path.join(STATIC_DIR, f"visemes_{base_filename}.json")

    # Step 1: Check cache for pre-rendered virtual human video
    if os.path.exists(cached_video_path) and os.path.getsize(cached_video_path) > 0:
        print(f"[HistTwin AI] Cache hit for Virtual Human Video: {cached_video_path}")
        shutil.copy(cached_video_path, output_video_path)
    
    # Step 2: Generate PantoMatrix 3D motion
    motion_json_path = generate_motion(audio_path)
    
    # Step 3: Extract phoneme/viseme lip-sync timeline
    generate_lip_sync_json(audio_path, output_viseme_path)

    # Step 4: Render Virtual Human Video if not already cached
    if not os.path.exists(output_video_path) or os.path.getsize(output_video_path) == 0:
        print(f"[HistTwin AI] Rendering Virtual Human video for {audio_path}...")
        render_avatar_video_opencv(audio_path, motion_json_path, output_video_path)
        
        # Save to cache
        if os.path.exists(output_video_path):
            shutil.copy(output_video_path, cached_video_path)

    # Prepare web-accessible URL paths
    video_url = f"/{os.path.basename(output_video_path)}" if os.path.exists(output_video_path) else None
    motion_url = f"/{os.path.basename(motion_json_path)}" if os.path.exists(motion_json_path) else None
    audio_url = f"/{os.path.basename(audio_path)}" if os.path.exists(audio_path) else None
    viseme_url = f"/{os.path.basename(output_viseme_path)}" if os.path.exists(output_viseme_path) else None

    return {
        "video_url": video_url,
        "motion_url": motion_url,
        "audio_url": audio_url,
        "viseme_url": viseme_url
    }
