import torch
import numpy as np
from .utils import resize_pixel_art, convert_to_grayscale, convert_to_bw
from .utils import PaletteGenerator, Dithering
import time

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
                    "max": 32,
                    "step": 1
                }),
                "scale_mode": (["auto", "nearest", "area", "linear", "cubic", "lanczos"],),
                "rescale_to_original": ("BOOLEAN", {"default": False}),
                "color_mode": (["rgb", "grayscale", "bw"],),
                "colors": ("INT", {
                    "default": 16,
                    "min": 2,
                    "max": 256,
                    "step": 1
                }),
                "quantization_method": (["auto", "kmeans", "mediancut", "maxcoverage", "fastoctree", "libimagequant", "median_cut"],),
                "dithering": (["none", "floyd-steinberg"],),
            },
            "optional": {
                "palette_image": ("IMAGE",),
                "palette_size": ("INT", {
                    "default": 32,
                    "min": 2,
                    "max": 256,
                    "step": 1
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "image/Pixel Art"

    def process(self, image, downscale_factor, scale_mode,
                rescale_to_original, color_mode, colors, 
                quantization_method, dithering, palette_image=None, palette_size=16):
        start_time = time.time()
        
        # Convert from torch tensor to numpy
        image_np = image[0].cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        
        print(f"Conversion to numpy: {time.time() - start_time:.2f}s   ")
        
        # Store original size if rescaling is needed
        original_size = (image_np.shape[1], image_np.shape[0]) if rescale_to_original else None
        
        # Apply pixel art scaling with new parameters
        t0 = time.time()
        image_np = resize_pixel_art(
            image_np,
            downscale_factor,
            rescale_to_original=rescale_to_original,
            original_size=original_size,
            scale_down_mode=scale_mode
        )
        print(f"Pixel art scaling ({scale_mode}): {time.time() - t0:.2f}s   ")
        
        # Apply color mode conversion
        t0 = time.time()
        if color_mode == "grayscale":
            image_np = convert_to_grayscale(image_np)
        elif color_mode == "bw":
            image_np = convert_to_bw(image_np)
        print(f"Color mode conversion: {time.time() - t0:.2f}s   ")
            
        # Get palette
        t0 = time.time()
        if palette_image is not None:
            # Convert palette image from torch tensor to numpy
            palette_np = palette_image[0].cpu().numpy()
            palette_np = (palette_np * 255).astype(np.uint8)
            palette = PaletteGenerator.get_palette(palette_np, palette_size)
        else:  # adaptive
            print(f"get_palette use {quantization_method}")
            palette = PaletteGenerator.get_palette(image_np, colors, quantization_method)

        print(f"Palette generation: {time.time() - t0:.2f}s   ")
                
        # Apply dithering if requested
        t0 = time.time()
        if dithering == "floyd-steinberg":
            image_np = Dithering.floyd_steinberg(image_np, palette)
        else:
            image_np = Dithering.simple_quantize(image_np, palette)
        print(f"Dithering: {time.time() - t0:.2f}s")
            
        # Convert back to torch tensor
        result = torch.from_numpy(image_np.astype(np.float32) / 255.0)
        result = result.unsqueeze(0)
        print(f"Conversion to tensor: {time.time() - t0:.2f}s   ")
        
        print(f"Total processing time: {time.time() - start_time:.2f}s   ")
        
        return (result,)