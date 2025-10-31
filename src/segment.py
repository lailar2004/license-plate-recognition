from ultralytics import YOLO
import cv2, os

model = YOLO("models/license_plate_detector.pt")

raw_dir = "data/raw"
seg_dir = "data/segmented"
color_dir = "data/color_classified"
os.makedirs(seg_dir, exist_ok=True)
os.makedirs(color_dir, exist_ok=True)

for file in os.listdir(raw_dir):
    if file.lower().endswith(('.jpg', '.png', '.jpeg')):
        raw_path = os.path.join(raw_dir, file)
        enhanced_path = os.path.join("data/enhanced", file)

        results = model(raw_path)
        image_color = cv2.imread(raw_path)
        image_gray = cv2.imread(enhanced_path, cv2.IMREAD_GRAYSCALE)

        for result in results:
            for i, box in enumerate(result.boxes.xyxy):
                x1, y1, x2, y2 = map(int, box)
                gray_crop = image_gray[y1:y2, x1:x2]
                color_crop = image_color[y1:y2, x1:x2]

                cv2.imwrite(f"{seg_dir}/{file[:-4]}_plate_{i}.jpg", gray_crop)
                cv2.imwrite(f"{color_dir}/{file[:-4]}_plate_{i}.jpg", color_crop)
