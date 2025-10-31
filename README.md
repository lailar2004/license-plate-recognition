License Plate Recognition System (India)

This project performs license plate detection, color classification, and text recognition for Indian vehicles using a pre-trained YOLOv8 model, OpenCV, and EasyOCR.
It is designed as a modular pipeline for computer vision–based vehicle identification.

📋 Features

Image Preprocessing – Enhances input images (grayscale conversion, histogram equalization, noise reduction, sharpening).

License Plate Detection – Uses a pre-trained YOLOv8 model to detect and crop license plates.

Color Classification – Classifies plate color (White, Yellow, Green, Red, Blue, or Black) using HSV color analysis.

Text Recognition (OCR) – Reads license plate numbers using EasyOCR.

Result Logging – Saves recognized text and plate color to a CSV file for further use.

🧩 Folder Structure
data/
 ├── raw/              # Input images
 ├── enhanced/         # Preprocessed images
 ├── segmented/        # Cropped license plates
 ├── color_classified/ # Plates analyzed for color
 └── results/          # Final output (CSV + logs)
src/
 ├── preprocess.py
 ├── segment.py
 ├── detect_plate_color.py
 ├── recognize_text.py
 └── main.py
models/
 └── license_plate_detector.pt   # Pretrained YOLOv8 model

⚙️ Setup

Install dependencies:

pip install ultralytics opencv-python easyocr numpy pandas matplotlib

▶️ How to Run

Run the entire pipeline:

python src/main.py


Each step (preprocess → detect → classify → OCR → save results) runs automatically.

🧠 Model Details

Model used: Pre-trained YOLOv8 license plate detector

Source: Public model from Roboflow Universe

No custom training performed

📊 Output Format

All results are saved in:

data/results/results.csv


Each row includes:

filename, recognized_text, plate_type
