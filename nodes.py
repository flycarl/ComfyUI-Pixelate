import torch
import numpy as np
from .utils import resize_pixel_art, convert_to_grayscale, convert_to_bw
from .utils import PaletteGenerator, Dithering

class ComfyUIPixelArtAdvanced:
    """
    Advanced Pixel Art node with multiple processing methods
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "downscale_factor": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 16,
                    "step": 1
                }),
                "rescale_to_original": ("BOOLEAN", {"default": False}),
                "color_mode": (["rgb", "grayscale", "bw"],),
                "colors": ("INT", {
                    "default": 16,
                    "min": 2,
                    "max": 256,
                    "step": 1
                }),
                "quantization_method": (["kmeans", "median_cut"],),
                "dithering": (["none", "floyd-steinberg"],),
                "palette_type": (["adaptive", "custom", "from_image"],),
            },
            "optional": {
                "custom_palette": ("STRING", {"default": "15,56,15;48,98,48;139,172,15;155,188,15"}),
                "palette_image": ("IMAGE",),
                "palette_size": ("INT", {
                    "default": 16,
                    "min": 2,
                    "max": 256,
                    "step": 1
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "image/Pixel Art"

    def process(self, image, downscale_factor, rescale_to_original, color_mode, colors, 
                quantization_method, dithering, palette_type, 
                custom_palette=None, palette_image=None, palette_size=16):
        # Convert from torch tensor to numpy
        image_np = image[0].cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        
        # Store original size if rescaling is needed
        original_size = (image_np.shape[1], image_np.shape[0]) if rescale_to_original else None
        
        # Apply pixel art scaling
        image_np = resize_pixel_art(image_np, downscale_factor, rescale_to_original, original_size)
        
        # Apply color mode conversion
        if color_mode == "grayscale":
            image_np = convert_to_grayscale(image_np)
        elif color_mode == "bw":
            image_np = convert_to_bw(image_np)
            
        # Get palette
        if palette_type == "custom" and custom_palette:
            palette = PaletteGenerator.parse_custom_palette(custom_palette)
            if palette is None:
                palette = PaletteGenerator.kmeans_palette(image_np, colors)
        elif palette_type == "from_image" and palette_image is not None:
            # Convert palette image from torch tensor to numpy
            palette_np = palette_image[0].cpu().numpy()
            palette_np = (palette_np * 255).astype(np.uint8)
            palette = PaletteGenerator.kmeans_palette(palette_np, palette_size)
        else:  # adaptive
            if quantization_method == "kmeans":
                palette = PaletteGenerator.kmeans_palette(image_np, colors)
            else:  # median_cut
                palette = PaletteGenerator.median_cut_palette(image_np, colors)
                
        # Apply dithering and quantization
        if dithering == "floyd-steinberg":
            result = Dithering.floyd_steinberg(image_np, palette)
        else:
            result = Dithering.simple_quantize(image_np, palette)
            
        # Convert back to torch tensor
        result = torch.from_numpy(result.astype(np.float32) / 255.0)
        result = result.unsqueeze(0)
        
        return (result,)