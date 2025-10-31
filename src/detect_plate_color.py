import cv2
import numpy as np
import os

color_dir = "data/color_classified"
for file in os.listdir(color_dir):
    path = os.path.join(color_dir, file)
    img = cv2.imread(path)
    if img is None:
        continue
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Use median instead of mean to avoid outliers
    med_hue = np.median(h)
    med_sat = np.median(s)
    med_val = np.median(v)
    
    
    # Classification tuned for OpenCV HSV (Hue: 0–179)
    # Check yellow first (most specific)
    if 15 < med_hue < 35 and med_sat > 80 and med_val > 100:
        plate_type = "Yellow - Commercial"
    # White plates have low saturation
    elif med_sat < 40 and med_val > 150:
        plate_type = "White - Private"
    # Green plates (includes cyan/turquoise shades)
    elif 40 < med_hue < 100 and med_sat > 60:
        plate_type = "Green - Electric Vehicle"
    # Blue plates (pure blue range)
    elif 100 < med_hue < 130 and med_sat > 80:
        plate_type = "Blue - Diplomatic"
    # Red plates (wraps around at 0/180)
    elif (med_hue < 10 or med_hue > 160) and med_sat > 80:
        plate_type = "Red - Government"
    # Black plates
    elif med_val < 80:
        plate_type = "Black - Commercial/Rental"
    else:
        plate_type = "Uncertain / Other"
    
    print(f"{file} -> {plate_type}")