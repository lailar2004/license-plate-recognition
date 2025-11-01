from ultralytics import YOLO
import cv2, os, sys

# --- Load YOLO model ---
model = YOLO("models/license_plate_detector.pt")

# --- Directories (one level up) ---
raw_dir = "data/raw"
enhanced_dir = "data/enhanced"
seg_dir = "data/segmented"
color_dir = "data/color_classified"

os.makedirs(seg_dir, exist_ok=True)
os.makedirs(color_dir, exist_ok=True)

# --- Check argument ---
if len(sys.argv) < 2:
    print("Usage: python detect_yolo.py <image_name>")
    sys.exit(1)

filename = sys.argv[1]

raw_path = os.path.join(raw_dir, filename)
enhanced_path = os.path.join(enhanced_dir, filename)

if not os.path.exists(raw_path):
    print(f"Error: {filename} not found in {raw_dir}")
    sys.exit(1)
if not os.path.exists(enhanced_path):
    print(f"Error: {filename} not found in {enhanced_dir}")
    sys.exit(1)

# --- Run YOLO detection ---
results = model(raw_path)

# --- Load images ---
image_color = cv2.imread(raw_path)
image_gray = cv2.imread(enhanced_path, cv2.IMREAD_GRAYSCALE)

if image_color is None or image_gray is None:
    print(f"Could not read images for {filename}")
    sys.exit(1)

# --- Extract and save detected plates ---
for result in results:
    for i, box in enumerate(result.boxes.xyxy):
        x1, y1, x2, y2 = map(int, box)
        gray_crop = image_gray[y1:y2, x1:x2]
        color_crop = image_color[y1:y2, x1:x2]

        gray_save_path = os.path.join(seg_dir, f"{filename}")
        color_save_path = os.path.join(color_dir, f"{filename}")

        cv2.imwrite(gray_save_path, gray_crop)
        cv2.imwrite(color_save_path, color_crop)

        print(f"Saved grayscale plate: {gray_save_path}")
        print(f"Saved color plate: {color_save_path}")

print(f"Detection completed for: {filename}")
