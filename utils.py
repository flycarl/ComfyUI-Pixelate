import numpy as np
import cv2
from PIL import Image
import colorsys

def scale_down(image, scale_factor):
    """
    Downscale image by integer factor using area interpolation
    
    Args:
        image: numpy array (H, W, C)
        scale_factor: int, factor to reduce image by
    """
    h, w = image.shape[:2]
    small_h, small_w = h // scale_factor, w // scale_factor
    return cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA)

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

def resize_pixel_art(image, scale_factor):
    """Resize image using pixel art scaling"""
    return scale_down(image, scale_factor)

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
    def kmeans_palette(image, n_colors):
        """Generate palette using k-means clustering"""
        pixels = image.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
        _, _, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        return palette

    @staticmethod
    def median_cut_palette(image, n_colors):
        """Generate palette using median cut algorithm"""
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

    @staticmethod
    def parse_custom_palette(palette_str):
        """Parse custom palette string in format 'R,G,B;R,G,B;...'"""
        try:
            return np.array([list(map(int, color.split(','))) 
                           for color in palette_str.split(';')])
        except:
            return None

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