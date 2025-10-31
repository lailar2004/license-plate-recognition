# verify_setup.py

import importlib
import os
import cv2
import pytesseract
import numpy as np
import matplotlib
import PIL

print("Project dependencies\n")

# List of core dependencies
packages = [
    "cv2", "numpy", "PIL", "matplotlib",
    "pytesseract", "skimage", "imutils", "pandas", "tqdm"
]

for pkg in packages:
    try:
        importlib.import_module(pkg.replace('-', '_'))
        print(f"\t{pkg} installed")
    except ImportError:
        print(f"{pkg} missing")

# Check Tesseract installation
try:
    tesseract_cmd = pytesseract.pytesseract.tesseract_cmd
    #print("\nTesseract path:", tesseract_cmd)

    version = pytesseract.get_tesseract_version()
    print("Tesseract version:", version)
except Exception as e:
    print("Tesseract not configured properly.")
    print(e)

# Check OpenCV basic function
try:
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print("\nOpenCV works fine")
except Exception as e:
    print("OpenCV test failed:", e)

# Check if project folders exist
required_dirs = [
    "data/raw", "data/preprocessed", "data/enhanced",
    "data/segmented", "data/color_classified", "data/results"
]

print("\nDestination folders")
for d in required_dirs:
    if os.path.exists(d):
        print(f"\t{d}")
    else:
        print(f"Missing folder: {d}")
