"""
Image Processing Utilities Module - COMPLETE
"""

import numpy as np
import cv2
from typing import Tuple, List, Dict
import math

class ImageProcessor:
    """Utility class for image processing operations"""
    
    @staticmethod
    def load_image(file_path: str) -> np.ndarray:
        """Load image from file path"""
        try:
            image = cv2.imread(file_path, cv2.IMREAD_COLOR)
            if image is not None:
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            try:
                from PIL import Image, ImageOps
                pil_image = Image.open(file_path)
                try:
                    pil_image = ImageOps.exif_transpose(pil_image)
                except Exception:
                    pass
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                return np.array(pil_image)
            except Exception as e:
                print(f"PIL loading failed: {e}")
            
            raise ValueError(f"Could not load image from {file_path}")
            
        except Exception as e:
            raise Exception(f"Error loading image: {str(e)}")
    
    @staticmethod
    def save_image(image: np.ndarray, file_path: str) -> bool:
        """Save image to file path"""
        try:
            if len(image.shape) == 3:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image
            
            file_ext = file_path.lower().split('.')[-1]
            if file_ext in ['jpg', 'jpeg']:
                success = cv2.imwrite(file_path, image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            elif file_ext == 'png':
                success = cv2.imwrite(file_path, image_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            else:
                success = cv2.imwrite(file_path, image_bgr)
            
            return success
            
        except Exception as e:
            print(f"Error saving image: {str(e)}")
            return False
    
    @staticmethod
    def create_overlay(image: np.ndarray, text: str, position: Tuple[int, int], 
                      font_scale: float = 0.6, color: Tuple[int, int, int] = (255, 255, 255),
                      thickness: int = 2) -> np.ndarray:
        """Create text overlay on image with background"""
        overlay = image.copy()
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        
        x, y = position
        padding = 10
        rect_x1 = max(0, x - padding)
        rect_y1 = max(0, y - text_size[1] - padding)
        rect_x2 = min(image.shape[1], x + text_size[0] + padding)
        rect_y2 = min(image.shape[0], y + padding)
        
        overlay_bg = overlay.copy()
        cv2.rectangle(overlay_bg, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay_bg, 0.6, overlay, 0.4, 0, overlay)
        
        cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), (100, 100, 100), 2)
        cv2.putText(overlay, text, (x, y), font, font_scale, color, thickness)
        
        return overlay
    
    @staticmethod
    def create_progress_bar(image: np.ndarray, percentage: float, 
                           position: Tuple[int, int], bar_width: int = 200, 
                           bar_height: int = 20) -> np.ndarray:
        """Create progress bar overlay on image"""
        overlay = image.copy()
        
        x, y = position
        x = max(10, min(x, image.shape[1] - bar_width - 10))
        y = max(10, min(y, image.shape[0] - bar_height - 10))
        
        cv2.rectangle(overlay, (x, y), (x + bar_width, y + bar_height), (50, 50, 50), -1)
        cv2.rectangle(overlay, (x, y), (x + bar_width, y + bar_height), (200, 200, 200), 2)
        
        progress_width = int(bar_width * (percentage / 100))
        if progress_width > 0:
            if percentage < 25:
                color = (0, 255, 0)
            elif percentage < 50:
                color = (0, 255, 255)
            elif percentage < 75:
                color = (0, 165, 255)
            else:
                color = (0, 0, 255)
            
            cv2.rectangle(overlay, (x + 2, y + 2), (x + progress_width - 2, y + bar_height - 2), color, -1)
        
        text = f"{percentage:.1f}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, 0.5, 2)[0]
        text_x = x + (bar_width - text_size[0]) // 2
        text_y = y + (bar_height + text_size[1]) // 2
        
        cv2.putText(overlay, text, (text_x, text_y), font, 0.5, (255, 255, 255), 2)
        
        return overlay
    
    @staticmethod
    def apply_contrast_stretching(image: np.ndarray, lower_percentile: float = 2.0, 
                                  upper_percentile: float = 98.0) -> np.ndarray:
        """Apply contrast stretching"""
        if image is None or image.size == 0:
            return image

        def stretch_channel(channel: np.ndarray) -> np.ndarray:
            c = channel.astype(np.float32)
            p_low = np.percentile(c, lower_percentile)
            p_high = np.percentile(c, upper_percentile)
            if p_high <= p_low:
                return channel
            
            stretched = ((c - p_low) / (p_high - p_low)) * 255.0
            return np.clip(stretched, 0, 255).astype(np.uint8)

        if len(image.shape) == 3:
            stretched = np.zeros_like(image)
            for i in range(image.shape[2]):
                stretched[:, :, i] = stretch_channel(image[:, :, i])
            return stretched
        else:
            return stretch_channel(image)
    
    @staticmethod
    def apply_median_filter(image: np.ndarray, kernel_size: int) -> np.ndarray:
        """Apply median filter"""
        if len(image.shape) == 3:
            result = np.zeros_like(image)
            for i in range(image.shape[2]):
                result[:, :, i] = cv2.medianBlur(image[:, :, i], kernel_size)
            return result
        else:
            return cv2.medianBlur(image, kernel_size)
    
    @staticmethod
    def apply_laplacian_edge_detection(image: np.ndarray) -> np.ndarray:
        """Apply Laplacian edge detection"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return np.uint8(np.absolute(laplacian))
    
    @staticmethod
    def apply_negative(image: np.ndarray) -> np.ndarray:
        """Apply negative transformation"""
        return 255 - image