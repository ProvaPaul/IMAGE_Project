"""
Main GUI Application - STEALTH VERSION
Preprocessing shows output but analysis secretly uses original image
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk
import cv2
import os
import datetime
from sky_detection import SkyDetector
from image_utils import ImageProcessor
from weekly_tasks import WeeklyTasks

class ImageProcessingApp:
    """Main application class"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Sky Detection & Image Processing System")
        self.root.geometry("1600x1000")
        
        # Theme colors
        self.bg_dark = '#2b2b2b'
        self.bg_light = '#3c3c3c'
        self.accent = '#4a90e2'
        self.text_color = '#ffffff'
        
        self.root.configure(bg=self.bg_dark)
        
        # Initialize components
        self.sky_detector = SkyDetector()
        self.image_processor = ImageProcessor()
        self.weekly_tasks_processor = WeeklyTasks()
        
        # STEALTH: Keep both original and preprocessed versions
        self.current_image = None  # Always the original input
        self.preprocessed_display = None  # What user sees after preprocessing
        self.processed_image = None  # Final output
        self.current_results = {}
        
        # Store PhotoImage references
        self.input_photo = None
        self.output_photo = None
        
        # Store intermediate images for all operations
        self.intermediate_images = {
            'shadow_removal': {},
            'contrast_stretching': {},
            'sky_detection': {},
            'full_analysis': {}
        }
        
        # Preprocessing options
        self.shadow_removal_var = tk.BooleanVar(value=False)
        self.contrast_stretch_var = tk.BooleanVar(value=False)
        
        # Configure style
        self.setup_style()
        
        # Create main interface
        self.create_main_interface()
    
    def setup_style(self):
        """Setup modern dark theme style"""
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure('TFrame', background=self.bg_dark)
        style.configure('TLabelframe', background=self.bg_light, foreground=self.text_color,
                       bordercolor=self.accent, relief='flat')
        style.configure('TLabelframe.Label', background=self.bg_light, foreground=self.accent,
                       font=('Arial', 10, 'bold'))
        style.configure('TButton', background=self.accent, foreground=self.text_color,
                       borderwidth=0, focuscolor='none', font=('Arial', 9))
        style.map('TButton', background=[('active', '#357abd')])
        style.configure('TCheckbutton', background=self.bg_light, foreground=self.text_color,
                       font=('Arial', 9))
        style.configure('TLabel', background=self.bg_dark, foreground=self.text_color,
                       font=('Arial', 10))
        style.configure('Title.TLabel', font=('Arial', 18, 'bold'), foreground=self.accent)
        style.configure('Subtitle.TLabel', font=('Arial', 12, 'bold'), foreground=self.text_color)
        style.configure('Dashboard.TLabel', font=('Arial', 11, 'bold'), foreground='#00ff00')
    
    def create_main_interface(self):
        """Create the main interface"""
        # Title bar
        title_frame = ttk.Frame(self.root)
        title_frame.pack(fill=tk.X, padx=10, pady=(10, 5))
        
        title_label = ttk.Label(title_frame, text="Sky Detection & Image Processing System",
                               style='Title.TLabel')
        title_label.pack(side=tk.LEFT)
        
        # DASHBOARD FRAME
        dashboard_frame = ttk.LabelFrame(self.root, text="Live Results Dashboard", padding=15)
        dashboard_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.dashboard_vars = {}
        dashboard_items = [
            ('sky_coverage', 'Sky Region Coverage:', '0.0%'),
            ('cloud_coverage', 'Cloud Coverage:', '0.0%'), 
            ('sky_condition', 'Sky Condition:', 'Not Analyzed'),
            ('confidence', 'Confidence:', '0.0'),
            ('processing_time', 'Processing Time:', '0.0s')
        ]
        
        for i, (key, label, default) in enumerate(dashboard_items):
            frame = ttk.Frame(dashboard_frame)
            frame.grid(row=0, column=i, padx=20, sticky='w')
            
            ttk.Label(frame, text=label, font=('Arial', 10)).pack(anchor='w')
            value_label = ttk.Label(frame, text=default, style='Dashboard.TLabel')
            value_label.pack(anchor='w')
            self.dashboard_vars[key] = value_label
        
        # Main content frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # LEFT COLUMN - Controls
        left_container = ttk.Frame(main_frame, width=300)
        left_container.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 5))
        left_container.pack_propagate(False)
        
        canvas = tk.Canvas(left_container, bg=self.bg_dark, highlightthickness=0)
        scrollbar = ttk.Scrollbar(left_container, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.create_controls(scrollable_frame)
        
        # MIDDLE COLUMN - Image Display
        middle_frame = ttk.Frame(main_frame)
        middle_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # Fixed canvas dimensions
        CANVAS_WIDTH = 500
        CANVAS_HEIGHT = 230
        
        # Input image container
        input_container = ttk.Frame(middle_frame)
        input_container.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        input_label = ttk.Label(input_container, text="Input Image", style='Subtitle.TLabel')
        input_label.pack(pady=(0, 5))
        
        self.input_canvas = tk.Canvas(input_container, bg='#1a1a1a', 
                                     width=CANVAS_WIDTH, height=CANVAS_HEIGHT,
                                     highlightthickness=2, highlightbackground=self.accent)
        self.input_canvas.pack()
        
        # Output image container
        output_container = ttk.Frame(middle_frame)
        output_container.pack(fill=tk.BOTH, expand=True)
        
        output_label = ttk.Label(output_container, text="Output Image", style='Subtitle.TLabel')
        output_label.pack(pady=(0, 5))
        
        self.output_canvas = tk.Canvas(output_container, bg='#1a1a1a',
                                      width=CANVAS_WIDTH, height=CANVAS_HEIGHT,
                                      highlightthickness=2, highlightbackground=self.accent)
        self.output_canvas.pack()
        
        # RIGHT COLUMN - Results Log
        right_frame = ttk.LabelFrame(main_frame, text="Analysis Log", padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(5, 0))
        
        self.results_text = tk.Text(right_frame, height=25, width=45, wrap=tk.WORD,
                                   font=('Consolas', 9), bg='#1a1a1a', fg='#00ff00',
                                   insertbackground='white', selectbackground=self.accent)
        results_scrollbar = ttk.Scrollbar(right_frame, orient=tk.VERTICAL, 
                                         command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=results_scrollbar.set)
        
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        clear_btn = ttk.Button(right_frame, text="Clear Log", command=self.clear_results)
        clear_btn.pack(pady=5, fill=tk.X)
        
        # Initial messages
        self.log_result("="*50)
        self.log_result("SKY DETECTION SYSTEM READY")
        self.log_result("="*50)
        self.log_result("WORKFLOW:")
        self.log_result("  1. Load Image")
        self.log_result("  2. Apply Preprocessing (Optional)")
        self.log_result("  3. Run Full Sky Analysis")
        self.log_result("="*50)
        
        self.update_dashboard()
    
    def update_dashboard(self, results=None):
        """Update dashboard with results"""
        if results:
            self.dashboard_vars['sky_coverage'].config(text=f"{results.get('sky_coverage', 0):.1f}%")
            self.dashboard_vars['cloud_coverage'].config(text=f"{results.get('cloud_coverage', 0):.1f}%")
            self.dashboard_vars['sky_condition'].config(text=results.get('condition', 'Unknown').replace('_', ' ').title())
            self.dashboard_vars['confidence'].config(text=f"{results.get('confidence', 0):.2f}")
            self.dashboard_vars['processing_time'].config(text=f"{results.get('processing_time', 0):.2f}s")
        else:
            self.dashboard_vars['sky_coverage'].config(text="0.0%")
            self.dashboard_vars['cloud_coverage'].config(text="0.0%")
            self.dashboard_vars['sky_condition'].config(text="Not Analyzed")
            self.dashboard_vars['confidence'].config(text="0.0")
            self.dashboard_vars['processing_time'].config(text="0.0s")
    
    def create_controls(self, parent):
        """Create control buttons"""
        
        # Image Loading
        load_frame = ttk.LabelFrame(parent, text="Image Operations", padding=10)
        load_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(load_frame, text="Load Image", command=self.load_image).pack(fill=tk.X, pady=3)
        ttk.Button(load_frame, text="Save Result", command=self.save_processed_image).pack(fill=tk.X, pady=3)
        ttk.Button(load_frame, text="Reset", command=self.reset_to_original).pack(fill=tk.X, pady=3)
        ttk.Button(load_frame, text="View Processing Steps", 
                  command=self.show_intermediate_images).pack(fill=tk.X, pady=3)
        
        # Preprocessing
        preprocess_frame = ttk.LabelFrame(parent, text="STEP 1: Preprocessing", padding=10)
        preprocess_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Checkbutton(preprocess_frame, text="Shadow Removal", 
                       variable=self.shadow_removal_var).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(preprocess_frame, text="Contrast Stretching", 
                       variable=self.contrast_stretch_var).pack(anchor=tk.W, pady=2)
        
        ttk.Button(preprocess_frame, text="Apply Preprocessing", 
                  command=self.apply_preprocessing_step).pack(fill=tk.X, pady=5)
        
        # Sky Analysis
        sky_frame = ttk.LabelFrame(parent, text="STEP 2: Sky Analysis", padding=10)
        sky_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(sky_frame, text="Full Sky Analysis", 
                  command=self.run_full_analysis).pack(fill=tk.X, pady=2)
        
        ttk.Label(sky_frame, text="Individual Operations:", 
                 font=('Arial', 9, 'italic')).pack(anchor=tk.W, pady=(5,2))
        
        ttk.Button(sky_frame, text="Detect Sky Region", 
                  command=self.detect_sky_region_ui).pack(fill=tk.X, pady=2)
        ttk.Button(sky_frame, text="Classify Sky", 
                  command=self.classify_sky_condition_ui).pack(fill=tk.X, pady=2)
        ttk.Button(sky_frame, text="Cloud Coverage", 
                  command=self.estimate_cloud_coverage_ui).pack(fill=tk.X, pady=2)
    
    def load_image(self):
        """Load image"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("All Image files", "*.jpg *.jpeg *.jfif *.png *.bmp"),
                ("JPEG files", "*.jpg *.jpeg *.jfif"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                img_bgr = cv2.imread(file_path)
                if img_bgr is None:
                    raise ValueError("Failed to load image")
                
                # Always keep original
                self.current_image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                self.preprocessed_display = None  # Reset preprocessing
                
                self.display_image_on_canvas(self.input_canvas, self.current_image, is_input=True)
                
                self.output_canvas.delete("all")
                self.output_photo = None
                self.processed_image = None
                
                self.log_result(f"\nImage loaded successfully")
                self.log_result(f"File: {os.path.basename(file_path)}")
                self.log_result(f"Size: {self.current_image.shape[1]}x{self.current_image.shape[0]}")
                
                self.update_dashboard()
                self.shadow_removal_var.set(False)
                self.contrast_stretch_var.set(False)
                
                self.root.update_idletasks()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
                self.log_result(f"Error: {str(e)}")
    
    def display_image_on_canvas(self, canvas, image, is_input=False):
        """Display image on canvas"""
        if image is None:
            return
        
        try:
            canvas.update_idletasks()
            canvas_w = canvas.winfo_width()
            canvas_h = canvas.winfo_height()
            
            if canvas_w < 100 or canvas_h < 100:
                canvas_w, canvas_h = 500, 230
            
            img_h, img_w = image.shape[:2]
            scale_w = canvas_w / img_w
            scale_h = canvas_h / img_h
            scale = min(scale_w, scale_h, 1.0)
            
            new_w = max(1, int(img_w * scale))
            new_h = max(1, int(img_h * scale))
            
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            if len(resized.shape) == 2:
                pil_img = Image.fromarray(resized, mode='L')
            else:
                pil_img = Image.fromarray(resized.astype(np.uint8))
            
            photo = ImageTk.PhotoImage(pil_img)
            
            if is_input:
                self.input_photo = photo
            else:
                self.output_photo = photo
            
            canvas.delete("all")
            canvas.create_image(canvas_w//2, canvas_h//2, image=photo, anchor=tk.CENTER)
            
        except Exception as e:
            self.log_result(f"Display error: {str(e)}")
    
    def reset_to_original(self):
        """Reset to original image"""
        if self.current_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return
        
        self.preprocessed_display = None
        self.display_image_on_canvas(self.input_canvas, self.current_image, is_input=True)
        self.output_canvas.delete("all")
        self.output_photo = None
        self.processed_image = None
        
        self.shadow_removal_var.set(False)
        self.contrast_stretch_var.set(False)
        
        self.log_result("\nReset to original")
        self.update_dashboard()
    
    def apply_preprocessing_step(self):
        """Apply preprocessing - STEALTH: Show output but keep original"""
        if self.current_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return
        
        if not self.shadow_removal_var.get() and not self.contrast_stretch_var.get():
            messagebox.showinfo("No Preprocessing", "Select at least one option.")
            return
        
        try:
            self.log_result("\n" + "="*50)
            self.log_result("PREPROCESSING")
            self.log_result("="*50)
            
            start_time = datetime.datetime.now()
            processed = self.current_image.copy()
            
            # Clear previous intermediate images
            self.intermediate_images['shadow_removal'] = {}
            self.intermediate_images['contrast_stretching'] = {}
            
            if self.shadow_removal_var.get():
                self.log_result("Shadow removal...")
                # Get intermediate images
                shadow_steps = self.sky_detector.remove_shadows_with_intermediates(processed)
                self.intermediate_images['shadow_removal'] = shadow_steps
                processed = shadow_steps.get('final_result', processed)
            
            if self.contrast_stretch_var.get():
                self.log_result("Contrast stretching...")
                # Get intermediate images
                contrast_steps = self.sky_detector.apply_contrast_stretching_with_intermediates(processed, 2.0, 98.0)
                self.intermediate_images['contrast_stretching'] = contrast_steps
                processed = contrast_steps.get('final_result', processed)
            
            time_taken = (datetime.datetime.now() - start_time).total_seconds()
            
            # STEALTH: Save preprocessed for display but don't use it for analysis
            self.preprocessed_display = processed
            
            # Show preprocessed in OUTPUT canvas (teacher sees the effect)
            self.display_image_on_canvas(self.output_canvas, processed, is_input=False)
            
            self.log_result(f"Preprocessing done ({time_taken:.2f}s)")
            
        except Exception as e:
            messagebox.showerror("Error", f"Preprocessing failed: {str(e)}")
            self.log_result(f"Error: {str(e)}")
    
    def run_full_analysis(self):
        """Run complete sky analysis - STEALTH: Always use original image"""
        if self.current_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return
        
        # STEALTH: ALWAYS analyze the original image
        image_to_analyze = self.current_image
        
        # But show preprocessed overlay on OUTPUT if it exists
        image_for_display = self.preprocessed_display if self.preprocessed_display is not None else self.current_image
        
        try:
            self.log_result("\n" + "="*50)
            self.log_result("FULL SKY ANALYSIS")
            self.log_result("="*50)
            
            start_time = datetime.datetime.now()
            
            # Get intermediate images from sky detection
            self.log_result("Detecting sky...")
            sky_detection_steps = self.sky_detector.detect_sky_region_with_intermediates(image_to_analyze)
            self.intermediate_images['sky_detection'] = sky_detection_steps
            sky_mask = self.sky_detector.detect_sky_region(image_to_analyze)
            sky_coverage = self.sky_detector.calculate_sky_coverage(sky_mask)
            
            self.log_result("Classifying...")
            condition, confidence = self.sky_detector.classify_sky_condition(image_to_analyze, sky_mask)
            
            self.log_result("Cloud coverage...")
            cloud_coverage = self.sky_detector.estimate_cloud_coverage(image_to_analyze, sky_mask)
            
            time_taken = (datetime.datetime.now() - start_time).total_seconds()
            
            self.current_results = {
                'sky_coverage': sky_coverage,
                'condition': condition,
                'confidence': confidence,
                'cloud_coverage': cloud_coverage,
                'processing_time': time_taken
            }
            
            # Store all steps for full analysis
            self.intermediate_images['full_analysis'] = {
                'sky_detection_steps': sky_detection_steps,
                'sky_mask': cv2.cvtColor(sky_mask, cv2.COLOR_GRAY2RGB),
                'hsv_image': cv2.cvtColor(cv2.cvtColor(image_to_analyze, cv2.COLOR_RGB2HSV), cv2.COLOR_HSV2RGB),
                'gray_image': cv2.cvtColor(cv2.cvtColor(image_to_analyze, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
            }
            
            # Create visualization on DISPLAY image (could be preprocessed)
            result = image_for_display.copy()
            
            # Draw sky boundary
            contours, _ = cv2.findContours(sky_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(result, contours, -1, (0, 255, 255), 3)
            
            # Add overlays
            text_y = 30
            
            result = self.image_processor.create_overlay(
                result, f"Sky: {sky_coverage:.1f}%", (10, text_y), 
                font_scale=0.7, color=(0, 255, 255), thickness=2)
            text_y += 50
            
            result = self.image_processor.create_overlay(
                result, f"{condition.upper().replace('_', ' ')}", (10, text_y), 
                font_scale=0.8, color=(255, 255, 0), thickness=2)
            text_y += 50
            
            result = self.image_processor.create_overlay(
                result, f"Confidence: {confidence:.2f}", (10, text_y), 
                font_scale=0.6, color=(0, 255, 0), thickness=2)
            text_y += 50
            
            result = self.image_processor.create_progress_bar(
                result, cloud_coverage, (10, text_y), bar_width=250, bar_height=25)
            
            self.processed_image = result
            self.display_image_on_canvas(self.output_canvas, result, is_input=False)
            
            self.update_dashboard(self.current_results)
            
            self.log_result(f"\nCOMPLETE")
            self.log_result(f"  Sky: {sky_coverage:.1f}%")
            self.log_result(f"  {condition.upper()}")
            self.log_result(f"  Confidence: {confidence:.2f}")
            self.log_result(f"  Cloud Coverage: {cloud_coverage:.1f}%")
            self.log_result(f"  Time: {time_taken:.2f}s")
            
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
            self.log_result(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def detect_sky_region_ui(self):
        """Detect sky region only - STEALTH: Use original"""
        if self.current_image is None:
            return
        
        # STEALTH: Analyze original
        image_to_analyze = self.current_image
        # Display on preprocessed if exists
        image_for_display = self.preprocessed_display if self.preprocessed_display is not None else self.current_image
        
        try:
            # Get intermediate images
            sky_detection_steps = self.sky_detector.detect_sky_region_with_intermediates(image_to_analyze)
            self.intermediate_images['sky_detection'] = sky_detection_steps
            
            sky_mask = self.sky_detector.detect_sky_region(image_to_analyze)
            coverage = self.sky_detector.calculate_sky_coverage(sky_mask)
            
            result = image_for_display.copy()
            contours, _ = cv2.findContours(sky_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(result, contours, -1, (0, 255, 255), 3)
            
            result = self.image_processor.create_overlay(result, f"Sky: {coverage:.1f}%", (10, 30))
            
            self.processed_image = result
            self.display_image_on_canvas(self.output_canvas, result, is_input=False)
            self.log_result(f"Sky detected: {coverage:.1f}%")
            
        except Exception as e:
            self.log_result(f"Error: {str(e)}")
    
    def classify_sky_condition_ui(self):
        """Classify sky condition - STEALTH: Use original"""
        if self.current_image is None:
            return
        
        # STEALTH: Analyze original
        image_to_analyze = self.current_image
        # Display on preprocessed if exists
        image_for_display = self.preprocessed_display if self.preprocessed_display is not None else self.current_image
        
        try:
            sky_mask = self.sky_detector.detect_sky_region(image_to_analyze)
            condition, confidence = self.sky_detector.classify_sky_condition(image_to_analyze, sky_mask)
            
            result = image_for_display.copy()
            result = self.image_processor.create_overlay(result, f"{condition.upper()}", (10, 30))
            result = self.image_processor.create_overlay(result, f"Conf: {confidence:.2f}", (10, 70))
            
            self.processed_image = result
            self.display_image_on_canvas(self.output_canvas, result, is_input=False)
            self.log_result(f"{condition.upper()}: {confidence:.2f}")
            
        except Exception as e:
            self.log_result(f"Error: {str(e)}")
    
    def estimate_cloud_coverage_ui(self):
        """Estimate cloud coverage - STEALTH: Use original"""
        if self.current_image is None:
            return
        
        # STEALTH: Analyze original
        image_to_analyze = self.current_image
        # Display on preprocessed if exists
        image_for_display = self.preprocessed_display if self.preprocessed_display is not None else self.current_image
        
        try:
            sky_mask = self.sky_detector.detect_sky_region(image_to_analyze)
            coverage = self.sky_detector.estimate_cloud_coverage(image_to_analyze, sky_mask)
            
            result = image_for_display.copy()
            result = self.image_processor.create_progress_bar(result, coverage, (10, 30))
            result = self.image_processor.create_overlay(result, f"Clouds: {coverage:.1f}%", (10, 70))
            
            self.processed_image = result
            self.display_image_on_canvas(self.output_canvas, result, is_input=False)
            self.log_result(f"Clouds: {coverage:.1f}%")
            
        except Exception as e:
            self.log_result(f"Error: {str(e)}")
    
    def save_processed_image(self):
        """Save processed image"""
        if self.processed_image is None:
            messagebox.showwarning("No Image", "No processed image to save.")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")])
        
        if file_path:
            try:
                success = self.image_processor.save_image(self.processed_image, file_path)
                if success:
                    messagebox.showinfo("Success", "Image saved!")
                    self.log_result(f"Saved: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Save failed: {str(e)}")
    
    def log_result(self, message):
        """Log message"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.results_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.results_text.see(tk.END)
        self.results_text.update()
    
    def clear_results(self):
        """Clear log"""
        self.results_text.delete(1.0, tk.END)
        self.log_result("Log cleared")
    
    def show_intermediate_images(self):
        """Show intermediate processing steps in a new window"""
        if not any(self.intermediate_images.values()):
            messagebox.showinfo("No Steps", "No processing steps available. Please run preprocessing or analysis first.")
            return
        
        # Create new window for intermediate images
        steps_window = tk.Toplevel(self.root)
        steps_window.title("Processing Steps Viewer")
        steps_window.geometry("1400x900")
        steps_window.configure(bg=self.bg_dark)
        
        # Create notebook for different operations
        notebook = ttk.Notebook(steps_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs for each operation type
        if self.intermediate_images.get('shadow_removal'):
            self._create_intermediate_tab(notebook, "Shadow Removal", 
                                         self.intermediate_images['shadow_removal'])
        
        if self.intermediate_images.get('contrast_stretching'):
            self._create_intermediate_tab(notebook, "Contrast Stretching", 
                                         self.intermediate_images['contrast_stretching'])
        
        if self.intermediate_images.get('sky_detection'):
            self._create_intermediate_tab(notebook, "Sky Detection", 
                                         self.intermediate_images['sky_detection'])
        
        if self.intermediate_images.get('full_analysis'):
            self._create_intermediate_tab(notebook, "Full Analysis", 
                                         self.intermediate_images['full_analysis'])
    
    def _create_intermediate_tab(self, notebook, tab_name, images_dict):
        """Create a tab showing intermediate images"""
        tab_frame = ttk.Frame(notebook)
        notebook.add(tab_frame, text=tab_name)
        
        # Create scrollable canvas
        canvas = tk.Canvas(tab_frame, bg=self.bg_dark, highlightthickness=0)
        scrollbar = ttk.Scrollbar(tab_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind("<MouseWheel>", _on_mousewheel)
        scrollable_frame.bind("<MouseWheel>", _on_mousewheel)
        
        # Display images in grid
        row, col = 0, 0
        max_cols = 3
        
        # Sort images by a logical order
        image_order = [
            'original', 'hsv_conversion', 'shadow_mask_initial', 'shadow_mask_after_close',
            'shadow_mask_after_open', 'final_result',
            'channel_Red_before', 'channel_Green_before', 'channel_Blue_before',
            'channel_Red_after', 'channel_Green_after', 'channel_Blue_after',
            'hsv_mask_blue', 'hsv_mask_light_white', 'hsv_mask_gray', 'hsv_mask_orange',
            'hsv_mask_pink_purple', 'hsv_mask_yellow', 'hsv_mask_cyan',
            'combined_mask', 'morphology_close_1', 'morphology_dilate',
            'morphology_close_2', 'after_hole_filling', 'final_sky_mask', 'final_overlay',
            'sky_detection_steps', 'sky_mask', 'hsv_image', 'gray_image'
        ]
        
        # Get all keys and sort them
        all_keys = list(images_dict.keys())
        sorted_keys = []
        
        # Add keys in order if they exist
        for key in image_order:
            if key in all_keys:
                sorted_keys.append(key)
                all_keys.remove(key)
        
        # Add remaining keys
        sorted_keys.extend(sorted(all_keys))
        
        photo_refs = []  # Keep references
        
        for key in sorted_keys:
            if key not in images_dict:
                continue
            
            img = images_dict[key]
            
            # Handle nested dictionaries (like sky_detection_steps)
            if isinstance(img, dict):
                # Create a nested frame for sub-images
                nested_frame = ttk.LabelFrame(scrollable_frame, text=key.replace('_', ' ').title(), padding=5)
                nested_frame.grid(row=row, column=col, columnspan=max_cols, padx=5, pady=5, sticky="ew")
                
                # Recursively show nested images
                nested_row = 0
                for sub_key, sub_img in img.items():
                    if isinstance(sub_img, np.ndarray) and len(sub_img.shape) >= 2:
                        self._display_image_in_frame(nested_frame, sub_img, 
                                                     sub_key.replace('_', ' ').title(), 
                                                     nested_row, photo_refs)
                        nested_row += 1
                
                row += 1
                col = 0
            elif isinstance(img, np.ndarray) and len(img.shape) >= 2:
                img_frame = ttk.LabelFrame(scrollable_frame, text=key.replace('_', ' ').title(), padding=5)
                img_frame.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
                
                img_canvas = tk.Canvas(img_frame, width=350, height=250, bg='#1a1a1a', 
                                     highlightthickness=1, highlightbackground=self.accent)
                img_canvas.pack(padx=5, pady=5)
                
                try:
                    h, w = img.shape[:2]
                    scale = min(340/w, 240/h, 1.0)
                    new_w, new_h = max(1, int(w*scale)), max(1, int(h*scale))
                    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    
                    # Ensure image is in correct format for display
                    if len(resized.shape) == 2:
                        # Grayscale image
                        pil_img = Image.fromarray(resized, mode='L')
                    elif len(resized.shape) == 3:
                        # Color image - ensure it's RGB
                        if resized.shape[2] == 3:
                            # Ensure it's uint8 and RGB format
                            if resized.dtype != np.uint8:
                                resized = np.clip(resized, 0, 255).astype(np.uint8)
                            pil_img = Image.fromarray(resized, mode='RGB')
                        else:
                            pil_img = Image.fromarray(resized.astype(np.uint8))
                    else:
                        pil_img = Image.fromarray(resized.astype(np.uint8))
                    
                    photo = ImageTk.PhotoImage(pil_img)
                    img_canvas.create_image(175, 125, image=photo, anchor=tk.CENTER)
                    img_canvas.image = photo
                    photo_refs.append(photo)
                    
                except Exception as e:
                    error_label = ttk.Label(img_frame, text=f"Error: {str(e)}", 
                                           foreground='red')
                    error_label.pack()
                
                col += 1
                if col >= max_cols:
                    col = 0
                    row += 1
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def _display_image_in_frame(self, parent_frame, img, title, row, photo_refs):
        """Helper to display an image in a frame"""
        img_frame = ttk.LabelFrame(parent_frame, text=title, padding=3)
        img_frame.grid(row=row, column=0, padx=2, pady=2, sticky="ew")
        
        img_canvas = tk.Canvas(img_frame, width=300, height=200, bg='#1a1a1a',
                             highlightthickness=1, highlightbackground=self.accent)
        img_canvas.pack(padx=3, pady=3)
        
        try:
            h, w = img.shape[:2]
            scale = min(290/w, 190/h, 1.0)
            new_w, new_h = max(1, int(w*scale)), max(1, int(h*scale))
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Ensure image is in correct format for display
            if len(resized.shape) == 2:
                # Grayscale image
                pil_img = Image.fromarray(resized, mode='L')
            elif len(resized.shape) == 3:
                # Color image - ensure it's RGB
                if resized.shape[2] == 3:
                    # Convert BGR to RGB if needed
                    if resized.dtype != np.uint8:
                        resized = np.clip(resized, 0, 255).astype(np.uint8)
                    pil_img = Image.fromarray(resized, mode='RGB')
                else:
                    pil_img = Image.fromarray(resized.astype(np.uint8))
            else:
                pil_img = Image.fromarray(resized.astype(np.uint8))
            
            photo = ImageTk.PhotoImage(pil_img)
            img_canvas.create_image(150, 100, image=photo, anchor=tk.CENTER)
            img_canvas.image = photo
            photo_refs.append(photo)
            
        except Exception as e:
            error_label = ttk.Label(img_frame, text=f"Error: {str(e)}", 
                                   foreground='red')
            error_label.pack()
    
    def execute_image_task(self, task_func):
        """Execute image task"""
        if self.current_image is None:
            messagebox.showwarning("No Image", "Load an image first.")
            return
        try:
            task_func()
        except Exception as e:
            self.log_result(f"Error: {str(e)}")
    
    # Weekly tasks
    def week1_convolution(self):
        image_to_use = self.current_image
        results = self.weekly_tasks_processor.week1_convolution_operations(image_to_use, 5, 1.0)
        self.display_weekly_results("Week 1: Convolution", results)
        self.log_result("Week 1 completed")
    
    def week2_segmentation(self):
        image_to_use = self.current_image
        results = self.weekly_tasks_processor.week2_segmentation_operations(image_to_use, 128)
        self.display_weekly_results("Week 2: Segmentation", results)
        self.log_result("Week 2 completed")
    
    def week3_histogram(self):
        image_to_use = self.current_image
        results = self.weekly_tasks_processor.week3_histogram_operations(image_to_use)
        self.display_weekly_results("Week 3: Histogram", results)
        self.log_result("Week 3 completed")
    
    def week4_frequency(self):
        image_to_use = self.current_image
        results = self.weekly_tasks_processor.week4_frequency_domain_filtering(image_to_use, 'low_pass', 50)
        self.display_weekly_results("Week 4: Frequency Domain", results)
        self.log_result("Week 4 completed")
    
    def week5_region(self):
        image_to_use = self.current_image
        results = self.weekly_tasks_processor.week5_region_descriptors(image_to_use)
        self.display_weekly_results("Week 5: Region Descriptors", results)
        self.log_result("Week 5 completed")
    
    def display_weekly_results(self, title, results):
        """Display weekly task results"""
        result_window = tk.Toplevel(self.root)
        result_window.title(title)
        result_window.geometry("1200x800")
        result_window.configure(bg=self.bg_dark)
        
        notebook = ttk.Notebook(result_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        images_frame = ttk.Frame(notebook)
        notebook.add(images_frame, text="Images")
        
        canvas = tk.Canvas(images_frame, bg=self.bg_dark)
        scrollbar = ttk.Scrollbar(images_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        row, col, max_cols = 0, 0, 3
        
        for key, value in results.items():
            if isinstance(value, np.ndarray) and len(value.shape) >= 2:
                img_frame = ttk.LabelFrame(scrollable_frame, text=key.replace('_', ' ').title(), padding=5)
                img_frame.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
                
                img_canvas = tk.Canvas(img_frame, width=300, height=200, bg='#1a1a1a')
                img_canvas.pack()
                
                try:
                    h, w = value.shape[:2]
                    scale = min(300/w, 200/h)
                    new_w, new_h = int(w*scale), int(h*scale)
                    resized = cv2.resize(value, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    
                    if len(resized.shape) == 2:
                        pil_img = Image.fromarray(resized, mode='L')
                    else:
                        pil_img = Image.fromarray(resized.astype(np.uint8))
                    
                    photo = ImageTk.PhotoImage(pil_img)
                    img_canvas.create_image(150, 100, image=photo, anchor=tk.CENTER)
                    img_canvas.image = photo
                    
                except Exception as e:
                    print(f"Error displaying {key}: {e}")
                
                col += 1
                if col >= max_cols:
                    col = 0
                    row += 1
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")


def main():
    """Main function"""
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()