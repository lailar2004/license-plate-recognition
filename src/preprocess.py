import cv2
import numpy as np
import os

# Input/output directories
input_dir = "data/raw"
enhanced_dir = "data/enhanced"
os.makedirs(enhanced_dir, exist_ok=True)

for file in os.listdir(input_dir):
    if file.lower().endswith(('.jpg', '.png', '.jpeg')):
        img_path = os.path.join(input_dir, file)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Could not read {file}")
            continue

        # --- 1️⃣ Denoise slightly (preserve edges)
        img_denoised = cv2.bilateralFilter(img, 9, 75, 75)

        # --- 2️⃣ Convert to LAB color space for better brightness/contrast control
        lab = cv2.cvtColor(img_denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # --- 3️⃣ Apply CLAHE (adaptive histogram equalization) on L-channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l_enhanced = clahe.apply(l)

        # --- 4️⃣ Merge and convert back to BGR
        lab_enhanced = cv2.merge((l_enhanced, a, b))
        enhanced_color = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        # --- 5️⃣ Sharpen slightly to emphasize edges
        sharpen_kernel = np.array([[0, -1, 0],
                                   [-1, 5, -1],
                                   [0, -1, 0]])
        enhanced_final = cv2.filter2D(enhanced_color, -1, sharpen_kernel)

        # --- 6️⃣ Save to enhanced directory
        save_path = os.path.join(enhanced_dir, file)
        cv2.imwrite(save_path, enhanced_final)
        print(f"Enhanced and saved: {save_path}")
