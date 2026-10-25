"""
HistTwin AI - Virtual Human Avatar System
Integrates PantoMatrix motion generation, audio lip-sync, and video rendering.
"""

from .motion_generator import generate_motion
from .lip_sync import extract_lip_sync, generate_lip_sync_json
from .avatar_renderer import generate_virtual_human_video

__all__ = [
    "generate_motion",
    "extract_lip_sync",
    "generate_lip_sync_json",
    "generate_virtual_human_video"
]
