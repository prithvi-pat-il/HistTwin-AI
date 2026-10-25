import os
import hashlib
import json
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STATIC_DIR = os.path.join(PROJECT_ROOT, "static")
CACHE_DIR = os.path.join(STATIC_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Standard Oculus / ARKit Viseme Shape Names
VISEMES = [
    "viseme_sil",  # Silence
    "viseme_PP",   # p, b, m
    "viseme_FF",   # f, v
    "viseme_TH",   # th
    "viseme_DD",   # d, t, n
    "viseme_kk",   # k, g
    "viseme_CH",   # ch, j, sh
    "viseme_SS",   # s, z
    "viseme_nn",   # n, l
    "viseme_RR",   # r
    "viseme_aa",   # ah, aa
    "viseme_E",    # eh, ey
    "viseme_I",    # ih, iy
    "viseme_O",    # ow, ao
    "viseme_U"     # uw, uh
]


def extract_lip_sync(audio_path: str, fps: int = 30) -> list[dict]:
    """
    Extracts acoustic energy and spectral features from an audio file 
    and maps them to 30 FPS facial visemes and mouth blendshapes (jawOpen, mouthOpen).
    
    Args:
        audio_path: Path to the input audio file (.wav or .mp3)
        fps: Target frame rate (default: 30)
        
    Returns:
        timeline: List of dictionary objects per frame containing viseme blendshapes.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Check cache
    hasher = hashlib.md5()
    with open(audio_path, "rb") as f:
        hasher.update(f.read())
    audio_hash = hasher.hexdigest()
    cache_file = os.path.join(CACHE_DIR, f"visemes_{audio_hash}.json")

    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            return json.load(f)

    timeline = []
    try:
        import librosa
        y, sr = librosa.load(audio_path, sr=16000)
        duration = librosa.get_duration(y=y, sr=sr)
        num_frames = int(duration * fps)
        
        # Calculate Short-Time Fourier Transform (STFT) for frequency analysis
        hop_length = int(sr / fps)
        stft = np.abs(librosa.stft(y, n_fft=512, hop_length=hop_length))
        
        # Spectral centroids and Root Mean Square (RMS) energy
        rms = librosa.feature.rms(y=y, frame_length=512, hop_length=hop_length)[0]
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
        
        # Normalize energy
        max_rms = np.max(rms) if np.max(rms) > 0 else 1.0
        normalized_rms = rms / max_rms

        for frame_idx in range(num_frames):
            if frame_idx < len(normalized_rms):
                energy = float(normalized_rms[frame_idx])
                centroid = float(spectral_centroids[frame_idx]) if frame_idx < len(spectral_centroids) else 1000.0
            else:
                energy = 0.0
                centroid = 0.0

            # Calculate primary mouth open / jaw open blendshapes
            mouth_open = float(np.clip(energy * 1.2, 0.0, 1.0))
            jaw_open = float(np.clip(energy * 0.85, 0.0, 0.85))

            # Select dominant viseme based on spectral characteristics
            if energy < 0.05:
                viseme = "viseme_sil"
                viseme_weights = {"viseme_sil": 1.0}
            elif centroid < 1200:
                viseme = "viseme_U"
                viseme_weights = {"viseme_U": min(1.0, energy * 1.3), "viseme_O": min(0.8, energy * 0.8)}
            elif centroid < 2200:
                viseme = "viseme_aa"
                viseme_weights = {"viseme_aa": min(1.0, energy * 1.4), "viseme_E": min(0.6, energy * 0.6)}
            elif centroid < 3500:
                viseme = "viseme_E"
                viseme_weights = {"viseme_E": min(1.0, energy * 1.3), "viseme_I": min(0.7, energy * 0.7)}
            else:
                viseme = "viseme_SS"
                viseme_weights = {"viseme_SS": min(1.0, energy * 1.2), "viseme_CH": min(0.5, energy * 0.5)}

            frame_data = {
                "frame": frame_idx,
                "time": round(frame_idx / fps, 3),
                "energy": round(energy, 3),
                "mouthOpen": round(mouth_open, 3),
                "jawOpen": round(jaw_open, 3),
                "dominantViseme": viseme,
                "visemes": {k: round(v, 3) for k, v in viseme_weights.items()}
            }
            timeline.append(frame_data)

    except Exception as e:
        print(f"[HistTwin AI] Procedural viseme fallback due to librosa exception: {e}")
        # Procedural timeline fallback
        num_frames = 150
        for frame_idx in range(num_frames):
            energy = 0.3 * (1.0 + np.sin(frame_idx * 0.5))
            timeline.append({
                "frame": frame_idx,
                "time": round(frame_idx / fps, 3),
                "energy": round(energy, 3),
                "mouthOpen": round(energy, 3),
                "jawOpen": round(energy * 0.7, 3),
                "dominantViseme": "viseme_aa",
                "visemes": {"viseme_aa": round(energy, 3)}
            })

    # Cache output
    with open(cache_file, "w") as f:
        json.dump(timeline, f, indent=2)

    return timeline


def generate_lip_sync_json(audio_path: str, output_json_path: str) -> str:
    """Generates and saves the viseme timeline to a JSON file."""
    timeline = extract_lip_sync(audio_path)
    with open(output_json_path, "w") as f:
        json.dump({"fps": 30, "frameCount": len(timeline), "timeline": timeline}, f, indent=2)
    return output_json_path
