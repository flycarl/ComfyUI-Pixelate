"""
Custom nodes for ComfyUI - Pixel Art Effects
"""
from .nodes import ComfyUIPixelate

NODE_CLASS_MAPPINGS = {
    "ComfyUIPixelate": ComfyUIPixelate
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyUIPixelate": "Pixelate"
} 