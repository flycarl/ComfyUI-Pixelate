import numpy as np
import cv2
from PIL import Image
import colorsys

def scale_down(image, scale_factor, mode='auto'):
    """
    Downscale image with advanced interpolation options
    
    Args:
        image: numpy array (H, W, C)
        scale_factor: int, factor to reduce image by
        mode: str, interpolation mode:
            - 'auto': Automatically select best method
            - 'area': cv2.INTER_AREA (good for downscaling)
            - 'nearest': cv2.INTER_NEAREST (preserves exact colors)
            - 'linear': cv2.INTER_LINEAR (smooth but can blur)
            - 'cubic': cv2.INTER_CUBIC (sharper than linear)
            - 'lanczos': cv2.INTER_LANCZOS4 (high quality, can preserve edges)
    """
    h, w = image.shape[:2]
    small_h, small_w = h // scale_factor, w // scale_factor

    # Dictionary of interpolation methods
    interpolation_methods = {
        'nearest': cv2.INTER_NEAREST,  # Best for pixel art
        'area': cv2.INTER_AREA,      # Good general downscaling
        'linear': cv2.INTER_LINEAR,   # Smooth, can blur
        'cubic': cv2.INTER_CUBIC,     # Sharper edges
        'lanczos': cv2.INTER_LANCZOS4 # High quality
    }
    if mode == 'auto':
        mode = 'area'

    # Apply selected interpolation
    return cv2.resize(image, (small_w, small_h), 
                     interpolation=interpolation_methods.get(mode, cv2.INTER_AREA))

def scale_up(image, scale_factor):
    """
    Upscale image by integer factor using nearest neighbor interpolation
    
    Args:
        image: numpy array (H, W, C)
        scale_factor: int, factor to enlarge image by
    """
    h, w = image.shape[:2]
    large_h, large_w = h * scale_factor, w * scale_factor
    return cv2.resize(image, (large_w, large_h), interpolation=cv2.INTER_NEAREST)

def resize_pixel_art(image, scale_factor, rescale_to_original=False, original_size=None, 
                    scale_down_mode='auto'):
    """
    Resize image using advanced pixel art scaling methods
    
    Args:
        image: numpy array (H, W, C)
        scale_factor: int, factor to reduce image by
        rescale_to_original: bool, whether to rescale back to original size
        original_size: tuple (width, height), original image dimensions
        scale_down_mode: str, interpolation mode:
            - 'auto': Automatically select best method
            - 'nearest': Best for pixel art
            - 'area': Good for general downscaling
            - 'linear': Smooth but can blur
            - 'cubic': Sharper edges
            - 'lanczos': High quality
    Returns:
        numpy array: Resized image
    """
    # Scale down first using advanced methods
    downscaled = scale_down(
        image, 
        scale_factor, 
        scale_down_mode
    )
    
    # If rescaling is requested and we have original size
    if rescale_to_original and original_size:
        h, w = original_size
        # Always use nearest neighbor for upscaling to maintain pixel art look
        return cv2.resize(downscaled, (w, h), interpolation=cv2.INTER_NEAREST)
    
    return downscaled

def convert_to_grayscale(image):
    """Convert image to grayscale while preserving dimensions"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

def convert_to_bw(image, threshold=127):
    """Convert image to black and white"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, bw = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(bw, cv2.COLOR_GRAY2RGB)

class PaletteGenerator:
    @staticmethod
    def pillow_palette(image, n_colors, method='libimagequant'):
        """Generate palette using Pillow's quantization methods
        
        Methods:
            - 'libimagequant': Highest quality, reasonable speed (default)
            - 'mediancut': Fast, good quality
            - 'maxcoverage': Better color coverage
            - 'fastoctree': Fastest, slightly lower quality
        """
        # Convert numpy array to PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Convert to RGB mode if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Quantize image
        if method == 'mediancut':
            quantized = image.quantize(colors=n_colors, method=Image.Quantize.MEDIANCUT)
        elif method == 'maxcoverage':
            quantized = image.quantize(colors=n_colors, method=Image.Quantize.MAXCOVERAGE)
        elif method == 'fastoctree':
            quantized = image.quantize(colors=n_colors, method=Image.Quantize.FASTOCTREE)
        else:  # libimagequant (default)
            quantized = image.quantize(colors=n_colors, method=Image.Quantize.LIBIMAGEQUANT)
            
        # Extract palette
        palette = np.array(quantized.getpalette()[:n_colors*3]).reshape(-1, 3)
        return palette

    @staticmethod
    def get_palette(image, n_colors, method='auto'):
        """Smart palette generation using multiple methods
        
        Methods:
            - 'auto': Choose best method based on image size and n_colors
            - 'libimagequant': Pillow's high quality quantizer
            - 'mediancut': Pillow's median cut
            - 'maxcoverage': Pillow's maximum coverage
            - 'fastoctree': Pillow's fast octree
            - 'kmeans': OpenCV k-means clustering
            - 'median_cut': Custom median cut implementation
        """
        # Auto method selection
        if method == 'auto':
            image_size = image.shape[0] * image.shape[1]
            if image_size > 1000000 or n_colors > 32:  # Large image or many colors
                method = 'fastoctree'
            elif image_size > 500000:  # Medium image
                method = 'libimagequant'
            else:  # Small image
                method = 'kmeans'

        print(f'Using method: {method} image size: {image.shape[0] * image.shape[1]}   ')
        # Use Pillow methods first
        if method in ['libimagequant', 'mediancut', 'maxcoverage', 'fastoctree']:
            return PaletteGenerator.pillow_palette(image, n_colors, method)
        
        # Fallback to OpenCV k-means
        elif method == 'kmeans':
            return PaletteGenerator.kmeans_palette(image, n_colors)
        
        # Custom median cut implementation
        elif method == 'median_cut':
            return PaletteGenerator.median_cut_palette(image, n_colors)
        
        else:
            raise ValueError(f"Unknown method: {method}")

    @staticmethod
    def kmeans_palette(image, n_colors):
        """Fallback k-means clustering method"""
        # Limit maximum colors for performance
        n_colors = min(n_colors, 32)
        pixels = image.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
        
        try:
            # Check if CUDA is available
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                # Move data to GPU
                gpu_mat = cv2.cuda_GpuMat()
                gpu_mat.upload(pixels)
                
                # Run k-means with GPU acceleration
                _, _, palette = cv2.kmeans(
                    gpu_mat.download(), 
                    n_colors, 
                    None, 
                    criteria, 
                    10, 
                    cv2.KMEANS_RANDOM_CENTERS + cv2.KMEANS_USE_GPU
                )
                print("Using GPU acceleration for k-means")
                return palette
                
        except Exception as e:
            print(f"GPU acceleration failed, falling back to CPU: {str(e)}")
        
        # CPU fallback with data reduction for better performance
        if len(pixels) > 10000:
            indices = np.random.choice(len(pixels), 10000, replace=False)
            pixels = pixels[indices]
            
        # Run k-means on CPU
        _, _, palette = cv2.kmeans(
            pixels, 
            n_colors, 
            None, 
            criteria, 
            10, 
            cv2.KMEANS_RANDOM_CENTERS
        )
        print("Using CPU for k-means")
        return palette

    @staticmethod
    def median_cut_palette(image, n_colors):
        """Fallback median cut implementation"""
        pixels = image.reshape(-1, 3)
        
        def cut_box(box_pixels):
            max_range = np.ptp(box_pixels, axis=0)
            cut_dim = np.argmax(max_range)
            box_pixels = box_pixels[box_pixels[:, cut_dim].argsort()]
            mid = len(box_pixels) // 2
            return box_pixels[:mid], box_pixels[mid:]

        boxes = [pixels]
        while len(boxes) < n_colors:
            box = boxes.pop(0)
            if len(box) < 2:
                break
            box1, box2 = cut_box(box)
            boxes.extend([box1, box2])

        return np.array([np.mean(box, axis=0) for box in boxes])
class Dithering:
    @staticmethod
    def extract_palette_from_image(image, palette_size=16):
        """Extract palette from reference image using k-means clustering"""
        pixels = image.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
        _, _, palette = cv2.kmeans(pixels, palette_size, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        return palette

    @staticmethod
    def find_closest_palette_color(pixel, palette):
        """Find the closest color in palette"""
        distances = np.sum((palette - pixel) ** 2, axis=1)
        return palette[np.argmin(distances)]

    @staticmethod
    def floyd_steinberg(image, palette=None, reference_image=None, palette_size=16):
        """
        Apply Floyd-Steinberg dithering
        
        Args:
            image: Image to dither
            palette: Optional pre-defined color palette
            reference_image: Optional reference image to extract palette from
            palette_size: Number of colors to extract from reference image
        """
        # Determine palette
        if palette is None and reference_image is not None:
            palette = Dithering.extract_palette_from_image(reference_image, palette_size)
        elif palette is None:
            # Fallback to simple palette if neither is provided
            palette = np.array([[0, 0, 0], [255, 255, 255]])

        height, width = image.shape[:2]
        result = image.copy().astype(np.float32)
        
        for y in range(height):
            for x in range(width):
                old_pixel = result[y, x].copy()
                new_pixel = Dithering.find_closest_palette_color(old_pixel, palette)
                result[y, x] = new_pixel
                
                quant_error = old_pixel - new_pixel
                
                if x + 1 < width:
                    result[y, x + 1] += quant_error * 7/16
                if y + 1 < height:
                    if x > 0:
                        result[y + 1, x - 1] += quant_error * 3/16
                    result[y + 1, x] += quant_error * 5/16
                    if x + 1 < width:
                        result[y + 1, x + 1] += quant_error * 1/16
        
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def simple_quantize(image, palette):
        """Simple color quantization using nearest neighbor"""
        height, width = image.shape[:2]
        result = np.zeros_like(image)
        color_cache = {}
        
        for y in range(height):
            for x in range(width):
                pixel = tuple(image[y, x])
                if pixel not in color_cache:
                    color_cache[pixel] = Dithering.find_closest_palette_color(image[y, x], palette)
                result[y, x] = color_cache[pixel]
        
        return result