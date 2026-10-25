import sys
import io
import os
import torch
import soundfile as sf

os.environ["PYTHONIOENCODING"] = "utf-8"
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


_model = None

def get_omnivoice_model():
    """
    Lazy loads the k2-fsa/OmniVoice model.
    Uses CUDA if available, otherwise falls back to CPU.
    """
    global _model
    if _model is None:
        from omnivoice import OmniVoice
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        print(f"Loading k2-fsa/OmniVoice TTS model on {device} ({dtype})...", flush=True)
        _model = OmniVoice.from_pretrained(
            "k2-fsa/OmniVoice",
            device_map=device,
            dtype=dtype
        )
        print("k2-fsa/OmniVoice model loaded successfully.", flush=True)
    return _model

# Valid OmniVoice English instruct tags: female, male, indian accent, young adult, low pitch, high pitch, etc.
LANGUAGE_INSTRUCT_MAP = {
    "English": "female, indian accent, young adult",
    "Hindi": "female, indian accent, young adult",
    "Marathi": "female, indian accent, young adult"
}

def synthesize_omnivoice(text: str, output_path: str, language: str = "English", instruct: str = None):
    """
    Synthesizes speech from text using k2-fsa/OmniVoice for English, Hindi, and Marathi.
    Saves audio to output_path (.wav format recommended).
    
    Args:
        text: The text history to synthesize into speech.
        output_path: File path to save output audio.
        language: Language of the text ("English", "Hindi", or "Marathi").
        instruct: Optional voice prompt/description overriding default language style.
    """
    model = get_omnivoice_model()
    
    if not instruct:
        instruct = LANGUAGE_INSTRUCT_MAP.get(language, LANGUAGE_INSTRUCT_MAP["English"])
        
    print(f"Synthesizing {language} speech with k2-fsa/OmniVoice (text length: {len(text)})...", flush=True)

    # Try generating speech with instruct prompt first, fallback to pure auto voice if instruct fails
    try:
        audio = model.generate(text=text, instruct=instruct)
    except Exception as ie:
        print(f"Instruct prompt '{instruct}' failed ({ie}). Generating with default auto voice...", flush=True)
        audio = model.generate(text=text)


    
    # Process audio output (tensor or array)
    if isinstance(audio, (list, tuple)):
        audio_tensor = audio[0]
    else:
        audio_tensor = audio
        
    if hasattr(audio_tensor, "cpu"):
        audio_data = audio_tensor.cpu().numpy()
    else:
        audio_data = audio_tensor
        
    # Ensure correct shape for soundfile (samples, channels)
    if len(audio_data.shape) > 1 and audio_data.shape[0] < audio_data.shape[1]:
        audio_data = audio_data.T
        
    # Save audio file (OmniVoice sampling rate is 24000 Hz)
    sf.write(output_path, audio_data, 24000)
    print(f"k2-fsa/OmniVoice speech audio saved to {output_path}")
    return output_path

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        text_arg = sys.argv[1]
        out_arg = sys.argv[2]
        lang_arg = sys.argv[3] if len(sys.argv) > 3 else "English"
        synthesize_omnivoice(text_arg, out_arg, lang_arg)

