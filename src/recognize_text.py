import easyocr
import cv2
import os
import re
import sys
import csv

# --- Initialize OCR reader ---
reader = easyocr.Reader(['en'], gpu=False)

# --- Directories (one level up) ---
input_dir = "data/segmented"
results_dir = "data/results"
os.makedirs(results_dir, exist_ok=True)

csv_path = os.path.join(results_dir, "recognize_text.csv")

# --- Check argument ---
if len(sys.argv) < 2:
    print("Usage: python ocr_read.py <image_name>")
    sys.exit(1)

filename = sys.argv[1]
input_path = os.path.join(input_dir, filename)

if not os.path.exists(input_path):
    print(f"Error: {filename} not found in {input_dir}")
    sys.exit(1)

def clean_plate_text(text):
    """Clean OCR output to match Indian plate format"""
    text = text.upper().replace(' ', '')
    text = re.sub(r'[^A-Z0-9]', '', text)

    # --- Fix common OCR mistakes ---
    if len(text) >= 2:
        replacements = {'0': 'O', '1': 'I', '6': 'G', '8': 'B', '5': 'S'}
        text = ''.join(replacements.get(c, c) if i < 2 and c.isdigit() else c for i, c in enumerate(text))

    if len(text) >= 4:
        replacements = {'O': '0', 'I': '1', 'Z': '2', 'S': '5', 'G': '6', 'B': '8'}
        text = text[:2] + ''.join(replacements.get(c, c) if 2 <= i < 4 and c.isalpha() else c for i, c in enumerate(text[2:], start=2))

    if len(text) >= 6:
        replacements = {'0': 'O', '1': 'I', '6': 'G', '8': 'B', '5': 'S', '2': 'Z'}
        text = text[:4] + ''.join(replacements.get(c, c) if 4 <= i < 6 and c.isdigit() else c for i, c in enumerate(text[4:], start=4))

    if len(text) >= 10:
        replacements = {'O': '0', 'I': '1', 'Z': '2', 'S': '5', 'G': '6', 'B': '8'}
        text = text[:-4] + ''.join(replacements.get(c, c) if c.isalpha() else c for c in text[-4:])

    # --- Pattern validation ---
    match = re.search(r'([A-Z]{2}\d{2}[A-Z]{1,2}\d{4})', text)
    if match:
        return match.group(1)
    if 8 <= len(text) <= 12:
        return text
    return text

# --- OCR Processing ---
img = cv2.imread(input_path)
if img is None:
    print(f"Could not read {filename}")
    sys.exit(1)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if contours:
    valid_contours = [c for c in contours if cv2.contourArea(c) > 50]
    if valid_contours:
        x_min = min([cv2.boundingRect(c)[0] for c in valid_contours])
        y_min = min([cv2.boundingRect(c)[1] for c in valid_contours])
        x_max = max([cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in valid_contours])
        y_max = max([cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in valid_contours])
        padding = 5
        gray = gray[
            max(0, y_min - padding):min(gray.shape[0], y_max + padding),
            max(0, x_min - padding):min(gray.shape[1], x_max + padding)
        ]

# --- Resize if needed ---
height, width = gray.shape[:2]
if width < 400:
    scale = 400 / width
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

gray = cv2.copyMakeBorder(gray, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=255)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# --- OCR ---
results = reader.readtext(thresh, detail=1, paragraph=False)
text = ''.join([r[1] for r in results]).replace(' ', '').upper() if results else "unreadable"
text = clean_plate_text(text) if text and text != "unreadable" else "unreadable"

# --- Print result ---
print(f"{filename} -> {text}")

# --- Save to CSV ---
file_exists = os.path.exists(csv_path)
with open(csv_path, mode='a', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    if not file_exists:
        writer.writerow(["filename", "recognize_text"])
    writer.writerow([filename, text])

print(f"Result saved to: {csv_path}")
