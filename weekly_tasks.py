"""
Weekly Tasks Implementation Module
"""

import numpy as np
import cv2
from typing import Tuple, List, Dict
from sky_detection import SkyDetector
from image_utils import ImageProcessor

class WeeklyTasks:
    """Implementation of weekly tasks"""
    
    def __init__(self):
        self.sky_detector = SkyDetector()
        self.image_processor = ImageProcessor()
    
    def week1_convolution_operations(self, image: np.ndarray, kernel_size: int = 5, 
                                   sigma: float = 1.0) -> Dict[str, np.ndarray]:
        """Week 1: Convolution Operations"""
        results = {}
        
        try:
            if image is None or image.size == 0:
                raise ValueError("Invalid input image")
            
            if image.dtype != np.float32:
                image_float = image.astype(np.float32) / 255.0
            else:
                image_float = image.copy()
            
            results['original'] = image.copy()
            results['gaussian_1'] = cv2.GaussianBlur(image_float, (kernel_size, kernel_size), 1.0)
            results['gaussian_2'] = cv2.GaussianBlur(image_float, (kernel_size, kernel_size), 2.0)
            results['sobel_x'] = cv2.Sobel(image_float, -1, 1, 0)
            results['sobel_y'] = cv2.Sobel(image_float, -1, 0, 1)
            results['laplacian'] = cv2.Laplacian(image_float, -1)
            
            for key, value in results.items():
                if isinstance(value, np.ndarray) and value.dtype != np.uint8:
                    if value.max() <= 1.0:
                        value = (value * 255).astype(np.uint8)
                    else:
                        value = np.clip(value, 0, 255).astype(np.uint8)
                    results[key] = value
            
            return results
            
        except Exception as e:
            print(f"Error in week1: {e}")
            return {'original': image.copy()}
    
    def week2_segmentation_operations(self, image: np.ndarray, threshold: int = 128) -> Dict[str, np.ndarray]:
        """Week 2: Segmentation"""
        results = {}
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        results['original'] = image.copy()
        results['grayscale'] = gray
        
        _, results['binary_thresh'] = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        _, results['otsu'] = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        results['canny'] = cv2.Canny(gray, 50, 150)
        
        return results
    
    def week3_histogram_operations(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Week 3: Histogram Operations"""
        results = {}
        results['original'] = image.copy()
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        results['grayscale'] = gray
        results['hist_equalized'] = cv2.equalizeHist(gray)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        results['clahe'] = clahe.apply(gray)
        
        return results
    
    def week4_frequency_domain_filtering(self, image: np.ndarray, filter_type: str = 'low_pass', 
                                       cutoff: int = 50) -> Dict[str, np.ndarray]:
        """Week 4: Frequency Domain Filtering"""
        results = {}
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        results['original'] = image.copy()
        results['grayscale'] = gray
        
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude_spectrum = np.log(np.abs(fft_shift) + 1)
        results['magnitude_spectrum'] = self._normalize_image(magnitude_spectrum)
        
        return results
    
    def week5_region_descriptors(self, image: np.ndarray) -> Dict[str, any]:
        """Week 5: Region Descriptors"""
        results = {}
        results['original'] = image.copy()
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        results['binary'] = binary
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        contour_image = image.copy()
        cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
        results['contour_image'] = contour_image
        
        return results
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to 0-255 range"""
        img_min = np.min(image)
        img_max = np.max(image)
        if img_max > img_min:
            return ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        return np.zeros_like(image, dtype=np.uint8)