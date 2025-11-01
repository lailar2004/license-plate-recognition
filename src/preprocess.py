import cv2
import numpy as np
import os
import sys

# --- Directories ---
input_dir = "data/raw"
enhanced_dir = "data/enhanced"

os.makedirs(enhanced_dir, exist_ok=True)

# --- Check for input argument ---
if len(sys.argv) < 2:
    print("Usage: python preprocess.py <image_name>")
    sys.exit(1)

filename = sys.argv[1]
input_path = os.path.join(input_dir, filename)

if not os.path.exists(input_path):
    print(f"Error: {filename} not found in {input_dir}")
    sys.exit(1)

# --- Read image ---
img = cv2.imread(input_path)
if img is None:
    print(f"Could not read image: {filename}")
    sys.exit(1)

# --- 1️⃣ Denoise (preserve edges) ---
img_denoised = cv2.bilateralFilter(img, 9, 75, 75)

# --- 2️⃣ Convert to LAB color space ---
lab = cv2.cvtColor(img_denoised, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)

# --- 3️⃣ Apply CLAHE on L-channel ---
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
l_enhanced = clahe.apply(l)

# --- 4️⃣ Merge and convert back to BGR ---
lab_enhanced = cv2.merge((l_enhanced, a, b))
enhanced_color = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

# --- 5️⃣ Sharpen image ---
sharpen_kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
enhanced_final = cv2.filter2D(enhanced_color, -1, sharpen_kernel)

# --- 6️⃣ Save output ---
save_path = os.path.join(enhanced_dir, filename)
cv2.imwrite(save_path, enhanced_final)

print(f"Enhanced and saved to: {save_path}")
