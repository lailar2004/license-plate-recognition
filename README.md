License Plate Recognition System (India)

> A complete image-processing pipeline for Indian license plate recognition using **YOLOv8**, **OpenCV**, and **EasyOCR**.

---

## Overview
This project detects, classifies, and reads vehicle license plates using a **pre-trained YOLOv8 model**.  
It performs:
- 🔧 **Preprocessing:** Image enhancement (contrast, noise reduction, sharpening)  
- 🎯 **Detection:** License plate localization using YOLOv8  
- 🎨 **Color Classification:** Identifies plate color (White, Yellow, Green, etc.)  
- 🔤 **OCR:** Extracts text using EasyOCR  
- 💾 **Result Logging:** Saves outputs to a CSV file  

---

## Project Structure
```bash
license_plate_recog_detec/
│
├── data/
│ ├── raw/ # Original images
│ ├── enhanced/ # Preprocessed grayscale images
│ ├── segmented/ # Cropped license plates
│ ├── color_classified/ # Plates analyzed for color
│ └── results/ # Final results (CSV + logs)
│
├── models/
│ └── license_plate_detector.pt # Pre-trained YOLOv8 model
│
└── src/
├── preprocess.py
├── segment.py
├── detect_plate_color.py
├── recognize_text.py
└── main.py


```

## Installation

```bash
# Clone this repository
git clone https://github.com/lailar2004/license-plate-recognition.git
cd license-plate-recognition
```
# Install dependencies
```bash
pip install ultralytics opencv-python easyocr numpy pandas matplotlib
```

Run the Pipeline

Run all steps automatically:

```bash
python src/main.py
```

This will:

Preprocess raw images

Detect and crop license plates

Classify plate color

Recognize text using OCR

Save all results to data/results/results.csv

Example Output
```bash
File	Detected Text	Plate Type
car1.jpg	KA01AB1234	White - Private
taxi2.jpg	MH12CD5678	Yellow - Commercial
```
Output file: data/results/results.csv

Model Details

Model: YOLOv8 (pre-trained)

Source: Roboflow Universe – License Plate Detection

Training: None (used as-is)