# 🌤️ SkyVision: Automated Sky Condition Analysis System

SkyVision is an image processing and computer vision application that automatically analyzes sky images and classifies weather conditions. The system can detect different sky types, estimate cloud coverage, and provide visual analysis through an intuitive graphical user interface.

Developed as a laboratory project for **CSE 4128: Image Processing and Computer Vision Laboratory** at **Khulna University of Engineering & Technology (KUET)**.

---

## 🎯 Objectives

* Detect sky regions automatically from images.
* Classify sky conditions:

  * Clear Sky
  * Cloudy Sky
  * Partly Cloudy Sky
  * Sunset Sky
* Estimate cloud coverage percentage.
* Apply image enhancement techniques for better analysis.
* Provide a user-friendly GUI for image processing and visualization.

---

## ✨ Features

* Sky segmentation using HSV color space.
* Multi-range color masking.
* Shadow removal preprocessing.
* Contrast stretching enhancement.
* Cloud coverage estimation.
* Confidence-based classification.
* Real-time visual dashboard.
* Export analysis results.
* Interactive desktop GUI using Tkinter.

---

## 🛠️ Technologies Used

### Programming Language

* Python 3.x

### Libraries

* OpenCV
* NumPy
* Pillow (PIL)
* Tkinter

### Development Tools

* VS Code / PyCharm
* Git & GitHub

---

## 📂 Project Structure

```bash
IMAGE_Project/
│
├── image_utils.py
├── sky_detection.py
├── main_gui_final.py
├── weekly_tasks.py
│
├── clear.jpeg
├── cloudy.jpeg
├── partly.jpeg
├── sunset.jpeg
│
├── images.jpg
├── project_report.pdf
├── 2007037.pptx
│
└── __pycache__/
```

---

## ⚙️ Methodology

### 1. Image Acquisition

* Load image.
* Convert image into RGB color space.

### 2. Preprocessing

* Shadow removal.
* Contrast stretching.
* Noise reduction.

### 3. Sky Detection

* HSV color space conversion.
* Seven different color masks.
* Morphological operations.
* Largest connected component extraction.

### 4. Feature Extraction

The system extracts:

* Blue ratio
* White ratio
* Gray ratio
* Warm color ratio
* Brightness
* Saturation
* Cloud coverage percentage

### 5. Classification

Weighted scoring is applied to classify:

* Clear
* Cloudy
* Partly Cloudy
* Sunset

along with confidence scores.

---

## 🖥️ User Interface

The application provides:

* Image upload
* Sky region visualization
* Cloud coverage analysis
* Classification confidence
* Dashboard metrics
* Export functionality

---

## 📊 Results

| Sky Condition | Accuracy | Average Confidence | Processing Time |
| ------------- | -------- | ------------------ | --------------- |
| Clear Sky     | 89%      | 0.87               | 1.2s            |
| Cloudy        | 82%      | 0.81               | 1.3s            |
| Partly Cloudy | 76%      | 0.74               | 1.1s            |
| Sunset        | 91%      | 0.89               | 1.2s            |

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/ProvaPaul/IMAGE_Project.git
```

Navigate to the project folder:

```bash
cd IMAGE_Project
```

Install dependencies:

```bash
pip install opencv-python numpy pillow
```

Run the application:

```bash
python main_gui_final.py
```

---

## 📸 Sample Outputs

The system successfully analyzes:

* Clear Sky Images
* Cloudy Sky Images
* Partly Cloudy Sky Images
* Sunset Sky Images

Add screenshots of your GUI and output results here.

---

## ⚠️ Limitations

* Partly cloudy classification remains challenging.
* Fixed HSV thresholds may not generalize to all environments.
* Processing speed is not suitable for real-time video applications.

---

## 🔮 Future Improvements

* Adaptive thresholding.
* Machine learning-based classification.
* Real-time video processing.
* Mobile application deployment.
* Weather forecasting integration.

---

## 👩‍💻 Author

**Prova Paul**
B.Sc. Engineering in Computer Science and Engineering
Khulna University of Engineering & Technology (KUET)

GitHub: https://github.com/ProvaPaul

---

## 📄 License

This project was developed for academic and educational purposes.
