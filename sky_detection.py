"""
Sky Detection Module - VERSION 8 (IMPROVED CLASSIFICATION)
Fixed: Better classification logic with weighted scoring system
"""

import numpy as np
import cv2
from typing import Tuple, Dict, Optional

class SkyDetector:
    """Main class for sky condition detection and classification"""
    
    def __init__(self):
        self.sky_conditions = {
            'clear': {'blue_threshold': 0.6, 'brightness_threshold': 0.7},
            'partly_cloudy': {'blue_threshold': 0.4, 'brightness_threshold': 0.5},
            'cloudy': {'blue_threshold': 0.2, 'brightness_threshold': 0.4}
        }
    
    def remove_shadows(self, image: np.ndarray, return_intermediates: bool = False) -> np.ndarray:
        """Remove shadows from image using inpainting"""
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            lower_shadow = np.array([0, 0, 0])
            upper_shadow = np.array([180, 255, 70])
            
            shadow_mask = cv2.inRange(hsv, lower_shadow, upper_shadow)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            shadow_mask_closed = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel)
            shadow_mask_final = cv2.morphologyEx(shadow_mask_closed, cv2.MORPH_OPEN, kernel)
            
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            inpainted_bgr = cv2.inpaint(image_bgr, shadow_mask_final, inpaintRadius=5, 
                                        flags=cv2.INPAINT_TELEA)
            result = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
            
            print("Shadow removal completed")
            return result
        except Exception as e:
            print(f"Shadow removal failed: {e}")
            return image
    
    def remove_shadows_with_intermediates(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Remove shadows and return intermediate steps"""
        try:
            steps = {}
            steps['original'] = image.copy()
            
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            steps['hsv_conversion'] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            
            lower_shadow = np.array([0, 0, 0])
            upper_shadow = np.array([180, 255, 70])
            shadow_mask = cv2.inRange(hsv, lower_shadow, upper_shadow)
            steps['shadow_mask_initial'] = cv2.cvtColor(shadow_mask, cv2.COLOR_GRAY2RGB)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            shadow_mask_closed = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel)
            steps['shadow_mask_after_close'] = cv2.cvtColor(shadow_mask_closed, cv2.COLOR_GRAY2RGB)
            
            shadow_mask_final = cv2.morphologyEx(shadow_mask_closed, cv2.MORPH_OPEN, kernel)
            steps['shadow_mask_after_open'] = cv2.cvtColor(shadow_mask_final, cv2.COLOR_GRAY2RGB)
            
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            inpainted_bgr = cv2.inpaint(image_bgr, shadow_mask_final, inpaintRadius=5, 
                                        flags=cv2.INPAINT_TELEA)
            result = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
            steps['final_result'] = result
            
            return steps
        except Exception as e:
            print(f"Shadow removal failed: {e}")
            return {'original': image.copy(), 'error': image.copy()}
    
    def apply_contrast_stretching(self, image: np.ndarray, 
                                  percentile_low: float = 2.0,
                                  percentile_high: float = 98.0) -> np.ndarray:
        """Apply contrast stretching using percentile-based method"""
        try:
            result = image.copy()
            
            for i in range(image.shape[2] if len(image.shape) == 3 else 1):
                if len(image.shape) == 3:
                    channel = image[:, :, i].astype(np.float32)
                else:
                    channel = image.astype(np.float32)
                
                R_min = np.percentile(channel, percentile_low)
                R_max = np.percentile(channel, percentile_high)
                
                if R_max - R_min < 1:
                    continue
                
                stretched = ((channel - R_min) / (R_max - R_min)) * 255.0
                stretched = np.clip(stretched, 0, 255).astype(np.uint8)
                
                if len(image.shape) == 3:
                    result[:, :, i] = stretched
                else:
                    result = stretched
            
            print(" Contrast stretching completed")
            return result
        except Exception as e:
            print(f"Contrast stretching failed: {e}")
            return image
    
    def apply_contrast_stretching_with_intermediates(self, image: np.ndarray,
                                                     percentile_low: float = 2.0,
                                                     percentile_high: float = 98.0) -> Dict[str, np.ndarray]:
        """Apply contrast stretching and return intermediate steps"""
        try:
            steps = {}
            steps['original'] = image.copy()
            
            result = image.copy()
            
            for i in range(image.shape[2] if len(image.shape) == 3 else 1):
                if len(image.shape) == 3:
                    channel = image[:, :, i].astype(np.float32)
                    channel_name = ['Red', 'Green', 'Blue'][i]
                else:
                    channel = image.astype(np.float32)
                    channel_name = 'Gray'
                
                R_min = np.percentile(channel, percentile_low)
                R_max = np.percentile(channel, percentile_high)
                
                if R_max - R_min < 1:
                    continue
                
                # Show histogram for this channel before stretching
                if len(image.shape) == 3:
                    channel_vis = image[:, :, i].copy()
                else:
                    channel_vis = image.copy()
                steps[f'channel_{channel_name}_before'] = cv2.cvtColor(channel_vis, cv2.COLOR_GRAY2RGB) if len(channel_vis.shape) == 2 else channel_vis
                
                stretched = ((channel - R_min) / (R_max - R_min)) * 255.0
                stretched = np.clip(stretched, 0, 255).astype(np.uint8)
                
                if len(image.shape) == 3:
                    result[:, :, i] = stretched
                    steps[f'channel_{channel_name}_after'] = cv2.cvtColor(stretched, cv2.COLOR_GRAY2RGB)
                else:
                    result = stretched
                    steps[f'channel_{channel_name}_after'] = cv2.cvtColor(stretched, cv2.COLOR_GRAY2RGB)
            
            steps['final_result'] = result
            return steps
        except Exception as e:
            print(f"Contrast stretching failed: {e}")
            return {'original': image.copy(), 'error': image.copy()}
    
    def preprocess_image(self, image: np.ndarray, 
                        remove_shadows: bool = True,
                        apply_contrast: bool = True) -> np.ndarray:
        """Comprehensive preprocessing pipeline"""
        processed = image.copy()
        
        if remove_shadows:
            processed = self.remove_shadows(processed)
        
        if apply_contrast:
            processed = self.apply_contrast_stretching(processed)
        
        return processed
    
    def detect_sky_region(self, image: np.ndarray) -> np.ndarray:
        """Detect sky region - Complete detection"""
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            height, width = image.shape[:2]
            
            masks = []
            
            # 1. Blue sky (EXPANDED for clear sky)
            lower_blue = np.array([85, 10, 30])
            upper_blue = np.array([145, 255, 255])
            mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
            masks.append(mask_blue)
            
            # 2. Light/white sky (clouds)
            lower_light = np.array([0, 0, 100])
            upper_light = np.array([180, 100, 255])
            mask_light = cv2.inRange(hsv, lower_light, upper_light)
            masks.append(mask_light)
            
            # 3. Gray sky (overcast/cloudy)
            lower_gray = np.array([0, 0, 50])
            upper_gray = np.array([180, 70, 220])
            mask_gray = cv2.inRange(hsv, lower_gray, upper_gray)
            masks.append(mask_gray)
            
            # 4. Orange sunset
            lower_orange = np.array([0, 20, 50])
            upper_orange = np.array([25, 255, 255])
            mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
            masks.append(mask_orange)
            
            # 5. Pink/purple sunset
            lower_pink = np.array([140, 20, 50])
            upper_pink = np.array([180, 255, 255])
            mask_pink = cv2.inRange(hsv, lower_pink, upper_pink)
            masks.append(mask_pink)
            
            # 6. Yellow
            lower_yellow = np.array([20, 20, 100])
            upper_yellow = np.array([40, 255, 255])
            mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
            masks.append(mask_yellow)
            
            # 7. Cyan/light blue
            lower_cyan = np.array([80, 10, 80])
            upper_cyan = np.array([100, 180, 255])
            mask_cyan = cv2.inRange(hsv, lower_cyan, upper_cyan)
            masks.append(mask_cyan)
            
            # Combine all masks
            sky_mask = np.zeros((height, width), dtype=np.uint8)
            for mask in masks:
                sky_mask = cv2.bitwise_or(sky_mask, mask)
            
            # Morphological operations
            kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
            kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            
            sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_CLOSE, kernel_large, iterations=3)
            sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_DILATE, kernel_medium, iterations=2)
            sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_CLOSE, kernel_large, iterations=2)
            
            # Fill holes
            contours, hierarchy = cv2.findContours(sky_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            for i, contour in enumerate(contours):
                if hierarchy[0][i][3] != -1:
                    cv2.drawContours(sky_mask, [contour], 0, 255, -1)
            
            # Keep largest component
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(sky_mask, connectivity=8)
            
            if num_labels > 1:
                largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                final_sky_mask = np.uint8(labels == largest_component) * 255
            else:
                final_sky_mask = sky_mask
            
            final_sky_mask = cv2.morphologyEx(final_sky_mask, cv2.MORPH_CLOSE, kernel_medium, iterations=1)
            
            sky_pixels = np.count_nonzero(final_sky_mask)
            total_pixels = height * width
            coverage = (sky_pixels / total_pixels) * 100
            
            print(f" Sky detection completed - Sky pixels: {sky_pixels} ({coverage:.1f}%)")
            return final_sky_mask
            
        except Exception as e:
            print(f"Sky detection failed: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    
    def detect_sky_region_with_intermediates(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Detect sky region and return all intermediate steps"""
        try:
            steps = {}
            steps['original'] = image.copy()
            
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            steps['hsv_conversion'] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            height, width = image.shape[:2]
            
            masks = []
            mask_names = ['blue', 'light_white', 'gray', 'orange', 'pink_purple', 'yellow', 'cyan']
            
            # 1. Blue sky
            lower_blue = np.array([85, 10, 30])
            upper_blue = np.array([145, 255, 255])
            mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
            masks.append(mask_blue)
            steps[f'hsv_mask_{mask_names[0]}'] = cv2.cvtColor(mask_blue, cv2.COLOR_GRAY2RGB)
            
            # 2. Light/white sky
            lower_light = np.array([0, 0, 100])
            upper_light = np.array([180, 100, 255])
            mask_light = cv2.inRange(hsv, lower_light, upper_light)
            masks.append(mask_light)
            steps[f'hsv_mask_{mask_names[1]}'] = cv2.cvtColor(mask_light, cv2.COLOR_GRAY2RGB)
            
            # 3. Gray sky
            lower_gray = np.array([0, 0, 50])
            upper_gray = np.array([180, 70, 220])
            mask_gray = cv2.inRange(hsv, lower_gray, upper_gray)
            masks.append(mask_gray)
            steps[f'hsv_mask_{mask_names[2]}'] = cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2RGB)
            
            # 4. Orange sunset
            lower_orange = np.array([0, 20, 50])
            upper_orange = np.array([25, 255, 255])
            mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
            masks.append(mask_orange)
            steps[f'hsv_mask_{mask_names[3]}'] = cv2.cvtColor(mask_orange, cv2.COLOR_GRAY2RGB)
            
            # 5. Pink/purple sunset
            lower_pink = np.array([140, 20, 50])
            upper_pink = np.array([180, 255, 255])
            mask_pink = cv2.inRange(hsv, lower_pink, upper_pink)
            masks.append(mask_pink)
            steps[f'hsv_mask_{mask_names[4]}'] = cv2.cvtColor(mask_pink, cv2.COLOR_GRAY2RGB)
            
            # 6. Yellow
            lower_yellow = np.array([20, 20, 100])
            upper_yellow = np.array([40, 255, 255])
            mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
            masks.append(mask_yellow)
            steps[f'hsv_mask_{mask_names[5]}'] = cv2.cvtColor(mask_yellow, cv2.COLOR_GRAY2RGB)
            
            # 7. Cyan/light blue
            lower_cyan = np.array([80, 10, 80])
            upper_cyan = np.array([100, 180, 255])
            mask_cyan = cv2.inRange(hsv, lower_cyan, upper_cyan)
            masks.append(mask_cyan)
            steps[f'hsv_mask_{mask_names[6]}'] = cv2.cvtColor(mask_cyan, cv2.COLOR_GRAY2RGB)
            
            # Combine all masks
            sky_mask = np.zeros((height, width), dtype=np.uint8)
            for mask in masks:
                sky_mask = cv2.bitwise_or(sky_mask, mask)
            steps['combined_mask'] = cv2.cvtColor(sky_mask, cv2.COLOR_GRAY2RGB)
            
            # Morphological operations
            kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
            kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            
            sky_mask_after_close1 = cv2.morphologyEx(sky_mask, cv2.MORPH_CLOSE, kernel_large, iterations=3)
            steps['morphology_close_1'] = cv2.cvtColor(sky_mask_after_close1, cv2.COLOR_GRAY2RGB)
            
            sky_mask_after_dilate = cv2.morphologyEx(sky_mask_after_close1, cv2.MORPH_DILATE, kernel_medium, iterations=2)
            steps['morphology_dilate'] = cv2.cvtColor(sky_mask_after_dilate, cv2.COLOR_GRAY2RGB)
            
            sky_mask_after_close2 = cv2.morphologyEx(sky_mask_after_dilate, cv2.MORPH_CLOSE, kernel_large, iterations=2)
            steps['morphology_close_2'] = cv2.cvtColor(sky_mask_after_close2, cv2.COLOR_GRAY2RGB)
            
            # Fill holes
            contours, hierarchy = cv2.findContours(sky_mask_after_close2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            for i, contour in enumerate(contours):
                if hierarchy[0][i][3] != -1:
                    cv2.drawContours(sky_mask_after_close2, [contour], 0, 255, -1)
            steps['after_hole_filling'] = cv2.cvtColor(sky_mask_after_close2, cv2.COLOR_GRAY2RGB)
            
            # Keep largest component
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(sky_mask_after_close2, connectivity=8)
            
            if num_labels > 1:
                largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                final_sky_mask = np.uint8(labels == largest_component) * 255
            else:
                final_sky_mask = sky_mask_after_close2
            
            final_sky_mask = cv2.morphologyEx(final_sky_mask, cv2.MORPH_CLOSE, kernel_medium, iterations=1)
            steps['final_sky_mask'] = cv2.cvtColor(final_sky_mask, cv2.COLOR_GRAY2RGB)
            
            # Overlay on original image
            overlay = image.copy()
            overlay[final_sky_mask > 0] = overlay[final_sky_mask > 0] * 0.7 + np.array([0, 255, 255]) * 0.3
            steps['final_overlay'] = overlay.astype(np.uint8)
            
            return steps
            
        except Exception as e:
            print(f"Sky detection failed: {e}")
            import traceback
            traceback.print_exc()
            return {'original': image.copy(), 'error': image.copy()}
    
    def calculate_sky_coverage(self, sky_mask: np.ndarray) -> float:
        """Calculate percentage of image covered by sky"""
        sky_pixels = np.count_nonzero(sky_mask)
        total_pixels = sky_mask.size
        
        if total_pixels == 0:
            return 0.0
        
        coverage = (sky_pixels / total_pixels) * 100
        return coverage
    
    def classify_sky_condition(self, image: np.ndarray, sky_mask: np.ndarray = None) -> Tuple[str, float]:
        """
        Classify sky condition - VERSION 8 IMPROVED
        Uses weighted scoring system for better accuracy
        """
        try:
            if sky_mask is None:
                sky_mask = self.detect_sky_region(image)
            
            sky_pixels_count = np.count_nonzero(sky_mask)
            if sky_pixels_count < 100:
                return 'unknown', 0.5
            
            sky_coverage = self.calculate_sky_coverage(sky_mask)
            
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            sky_pixels_rgb = image[sky_mask > 0]
            sky_pixels_hsv = hsv[sky_mask > 0]
            sky_pixels_gray = gray[sky_mask > 0]
            
            if len(sky_pixels_rgb) == 0:
                return 'unknown', 0.5
            
            # Calculate color features
            blue_ratio = self._calculate_blue_ratio(sky_pixels_hsv)
            white_ratio = self._calculate_white_ratio(sky_pixels_hsv)
            gray_ratio = self._calculate_gray_ratio(sky_pixels_hsv)
            warm_ratio = self._calculate_warm_ratio(sky_pixels_hsv)
            orange_ratio = self._calculate_orange_ratio(sky_pixels_hsv)
            
            # Calculate brightness and saturation
            mean_brightness = np.mean(sky_pixels_gray) / 255.0
            std_brightness = np.std(sky_pixels_gray) / 255.0
            mean_saturation = np.mean(sky_pixels_hsv[:, 1]) / 255.0
            
            # Calculate RGB means
            mean_r = np.mean(sky_pixels_rgb[:, 0]) / 255.0
            mean_g = np.mean(sky_pixels_rgb[:, 1]) / 255.0
            mean_b = np.mean(sky_pixels_rgb[:, 2]) / 255.0
            
            # Cloud coverage calculation
            cloud_threshold = np.mean(sky_pixels_gray) + np.std(sky_pixels_gray) * 0.3
            bright_pixels = np.sum(sky_pixels_gray > cloud_threshold)
            cloud_ratio = bright_pixels / len(sky_pixels_gray) if len(sky_pixels_gray) > 0 else 0
            
            # **DEBUG OUTPUT**
            print(f"\n DETAILED SKY ANALYSIS:")
            print(f"=" * 60)
            print(f"Blue Ratio:     {blue_ratio:.3f}")
            print(f"White Ratio:    {white_ratio:.3f}")
            print(f"Gray Ratio:     {gray_ratio:.3f}")
            print(f"Warm Ratio:     {warm_ratio:.3f}")
            print(f"Orange Ratio:   {orange_ratio:.3f}")
            print(f"Cloud Ratio:    {cloud_ratio:.3f}")
            print(f"Brightness:     {mean_brightness:.3f}")
            print(f"Saturation:     {mean_saturation:.3f}")
            print(f"RGB Balance:    R={mean_r:.2f} G={mean_g:.2f} B={mean_b:.2f}")
            print(f"=" * 60)
            
            # **WEIGHTED SCORING SYSTEM**
            scores = {
                'clear': 0.0,
                'partly_cloudy': 0.0, 
                'cloudy': 0.0,
                'sunset': 0.0
            }
            
            # **1. CLEAR SKY SCORING**
            if blue_ratio > 0.25:
                scores['clear'] += blue_ratio * 2.0
            if mean_brightness > 0.6:
                scores['clear'] += (mean_brightness - 0.6) * 1.5
            if mean_saturation > 0.2:
                scores['clear'] += (mean_saturation - 0.2) * 1.0
            if cloud_ratio < 0.25:
                scores['clear'] += (0.25 - cloud_ratio) * 2.0
            
            # **2. CLOUDY SCORING** 
            if white_ratio > 0.2:
                scores['cloudy'] += white_ratio * 2.0
            if gray_ratio > 0.2:
                scores['cloudy'] += gray_ratio * 2.0
            if cloud_ratio > 0.4:
                scores['cloudy'] += (cloud_ratio - 0.4) * 1.5
            if blue_ratio < 0.15:  # Low blue = more likely cloudy
                scores['cloudy'] += (0.15 - blue_ratio) * 1.0
            if mean_brightness < 0.6:  # Darker = more likely cloudy
                scores['cloudy'] += (0.6 - mean_brightness) * 1.0
            
            # **3. PARTLY CLOUDY SCORING**
            if 0.15 <= blue_ratio <= 0.35:
                scores['partly_cloudy'] += blue_ratio * 1.5
            if 0.2 <= cloud_ratio <= 0.5:
                scores['partly_cloudy'] += cloud_ratio * 1.0
            if 0.15 <= white_ratio <= 0.35:
                scores['partly_cloudy'] += white_ratio * 1.0
            
            # **4. SUNSET SCORING**
            if warm_ratio > 0.1:
                scores['sunset'] += warm_ratio * 3.0
            if orange_ratio > 0.08:
                scores['sunset'] += orange_ratio * 3.0
            if mean_r > mean_b * 1.2:  # Red > Blue
                scores['sunset'] += (mean_r - mean_b) * 2.0
            if mean_saturation > 0.3:  # High saturation in sunset
                scores['sunset'] += (mean_saturation - 0.3) * 1.0
            
            # **PENALTIES** (conditions that make a classification unlikely)
            # Clear sky shouldn't have high white/gray
            if white_ratio > 0.3 or gray_ratio > 0.3:
                scores['clear'] *= 0.3
            
            # Cloudy shouldn't have high blue
            if blue_ratio > 0.25:
                scores['cloudy'] *= 0.4
            
            # Partly cloudy needs some blue AND some clouds
            if blue_ratio < 0.1 or cloud_ratio < 0.1:
                scores['partly_cloudy'] *= 0.3
            
            # Sunset shouldn't have high blue
            if blue_ratio > 0.2:
                scores['sunset'] *= 0.4
            
            # **DISPLAY SCORES**
            print(f"\n CLASSIFICATION SCORES:")
            for condition, score in scores.items():
                print(f"  {condition.upper():<15}: {score:.3f}")
            
            # **FINAL CLASSIFICATION**
            best_condition = max(scores, key=scores.get)
            best_score = scores[best_condition]
            
            # Calculate confidence based on score dominance
            total_score = sum(scores.values())
            if total_score > 0:
                confidence = min(0.95, best_score / total_score * 2)
            else:
                confidence = 0.5
            
            # **ENSURE MINIMUM THRESHOLDS**
            if best_score < 0.1:  # Very low confidence
                # Fallback to simple rules
                if blue_ratio > 0.3 and cloud_ratio < 0.2:
                    best_condition = 'clear'
                    confidence = 0.7
                elif (white_ratio > 0.3 or gray_ratio > 0.3) and blue_ratio < 0.15:
                    best_condition = 'cloudy' 
                    confidence = 0.7
                elif warm_ratio > 0.1:
                    best_condition = 'sunset'
                    confidence = 0.7
                else:
                    best_condition = 'partly_cloudy'
                    confidence = 0.6
            
            print(f"\n FINAL: {best_condition.upper()} (confidence: {confidence:.2f})")
            return best_condition, confidence
            
        except Exception as e:
            print(f" Classification error: {e}")
            import traceback
            traceback.print_exc()
            return 'unknown', 0.5
    
    def _calculate_blue_ratio(self, hsv_pixels: np.ndarray) -> float:
        """Calculate ratio of blue pixels"""
        if len(hsv_pixels) == 0:
            return 0.0
        mask = ((hsv_pixels[:, 0] >= 85) & (hsv_pixels[:, 0] <= 145) & 
                (hsv_pixels[:, 1] > 10) & (hsv_pixels[:, 2] > 30))
        return np.sum(mask) / len(hsv_pixels)
    
    def _calculate_white_ratio(self, hsv_pixels: np.ndarray) -> float:
        """Calculate ratio of white/bright pixels"""
        if len(hsv_pixels) == 0:
            return 0.0
        mask = (hsv_pixels[:, 1] <= 80) & (hsv_pixels[:, 2] >= 120)
        return np.sum(mask) / len(hsv_pixels)
    
    def _calculate_gray_ratio(self, hsv_pixels: np.ndarray) -> float:
        """Calculate ratio of gray pixels"""
        if len(hsv_pixels) == 0:
            return 0.0
        mask = (hsv_pixels[:, 1] <= 60) & (hsv_pixels[:, 2] >= 60) & (hsv_pixels[:, 2] <= 200)
        return np.sum(mask) / len(hsv_pixels)
    
    def _calculate_warm_ratio(self, hsv_pixels: np.ndarray) -> float:
        """Calculate ratio of warm colored pixels"""
        if len(hsv_pixels) == 0:
            return 0.0
        mask = (((hsv_pixels[:, 0] <= 25) | (hsv_pixels[:, 0] >= 140)) & 
                (hsv_pixels[:, 1] > 20) & (hsv_pixels[:, 2] > 50))
        return np.sum(mask) / len(hsv_pixels)
    
    def _calculate_orange_ratio(self, hsv_pixels: np.ndarray) -> float:
        """Calculate ratio of orange pixels"""
        if len(hsv_pixels) == 0:
            return 0.0
        mask = ((hsv_pixels[:, 0] >= 0) & (hsv_pixels[:, 0] <= 25) & 
                (hsv_pixels[:, 1] > 50) & (hsv_pixels[:, 2] > 100))
        return np.sum(mask) / len(hsv_pixels)
    
    def estimate_cloud_coverage(self, image: np.ndarray, sky_mask: np.ndarray = None) -> float:
        """Estimate cloud coverage percentage"""
        try:
            if sky_mask is None:
                sky_mask = self.detect_sky_region(image)
            
            if np.count_nonzero(sky_mask) < 100:
                return 0.0
            
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            sky_gray = gray[sky_mask > 0]
            
            mean_brightness = np.mean(sky_gray)
            std_brightness = np.std(sky_gray)
            
            cloud_threshold = mean_brightness + std_brightness * 0.5
            
            cloud_pixels = np.sum(sky_gray > cloud_threshold)
            sky_pixels = len(sky_gray)
            
            coverage = (cloud_pixels / sky_pixels) * 100
            
            return min(100, max(0, coverage))
            
        except Exception as e:
            print(f"Cloud coverage estimation failed: {e}")
            return 0.0