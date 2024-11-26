import torch
import numpy as np
from .utils import resize_pixel_art, convert_to_grayscale, convert_to_bw
from .utils import PaletteGenerator, Dithering
import time

class ComfyUIPixelate:
    """
    Scale Down and Pixelate image
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
        
        # Separate RGB and Alpha channels
        has_alpha = image_np.shape[-1] == 4
        if has_alpha:
            rgb_channels = image_np[:, :, :3]
            alpha_channel = image_np[:, :, 3]
        else:
            rgb_channels = image_np
        
        # Store original size if rescaling is needed
        original_size = (rgb_channels.shape[1], rgb_channels.shape[0]) if rescale_to_original else None
        
        # Apply pixel art scaling with new parameters
        rgb_channels = resize_pixel_art(
            rgb_channels,
            downscale_factor,
            rescale_to_original=rescale_to_original,
            original_size=original_size,
            scale_down_mode=scale_mode
        )
        
        if has_alpha:
            alpha_channel = resize_pixel_art(
                alpha_channel[..., np.newaxis],
                downscale_factor,
                rescale_to_original=rescale_to_original,
                original_size=original_size,
                scale_down_mode=scale_mode
            )
        
        # Apply color mode conversion
        if color_mode == "grayscale":
            rgb_channels = convert_to_grayscale(rgb_channels)
        elif color_mode == "bw":
            rgb_channels = convert_to_bw(rgb_channels)
            
        # Get palette
        if palette_image is not None:
            palette_np = palette_image[0].cpu().numpy()
            palette_np = (palette_np * 255).astype(np.uint8)
            palette = PaletteGenerator.get_palette(palette_np[:, :, :3], palette_size)
        else:
            palette = PaletteGenerator.get_palette(rgb_channels, colors, quantization_method)
                
        # Apply dithering if requested
        if dithering == "floyd-steinberg":
            rgb_channels = Dithering.floyd_steinberg(rgb_channels, palette)
        else:
            rgb_channels = Dithering.simple_quantize(rgb_channels, palette)
        
        # Recombine RGB and Alpha channels
        if has_alpha:
            image_np = np.dstack((rgb_channels, alpha_channel))
        else:
            image_np = rgb_channels
            
        # Convert back to torch tensor
        result = torch.from_numpy(image_np.astype(np.float32) / 255.0)
        result = result.unsqueeze(0)
        
        return (result,)