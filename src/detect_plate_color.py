import cv2
import numpy as np
import os
import sys
import pandas as pd

# --- Directories ---
color_dir = "data/color_classified"
results_path = "data/results/detect_plate_color.csv"

# --- Check argument ---
if len(sys.argv) < 2:
    print("Usage: python detect_plate_color.py <image_name>")
    sys.exit(1)

filename = sys.argv[1]
input_path = os.path.join(color_dir, filename)

if not os.path.exists(input_path):
    print(f"Error: {filename} not found in {color_dir}")
    sys.exit(1)

# --- Read image ---
img = cv2.imread(input_path)
if img is None:
    print(f"Could not read {filename}")
    sys.exit(1)

# --- Convert to HSV ---
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

# --- Median values to reduce noise ---
med_hue = np.median(h)
med_sat = np.median(s)
med_val = np.median(v)

# --- Improved Classification Logic ---
# White plates: Low saturation, high value
if med_sat < 60 and med_val > 120:
    plate_type = "White - Private"
# Yellow plates: Hue in yellow range, moderate-high saturation
elif 15 < med_hue < 35 and med_sat > 60 and med_val > 100:
    plate_type = "Yellow - Commercial"
# Green plates: Hue in green range
elif 35 < med_hue < 85 and med_sat > 40:
    plate_type = "Green - Electric Vehicle"
# Blue plates: Hue in blue range
elif 90 < med_hue < 130 and med_sat > 50:
    plate_type = "Blue - Diplomatic"
# Red plates: Hue in red range (wraps around at 0/180)
elif (med_hue < 10 or med_hue > 160) and med_sat > 60:
    plate_type = "Red - Government"
# Black plates: Very low value
elif med_val < 70:
    plate_type = "Black - Commercial/Rental"
else:
    plate_type = "Uncertain / Other"

# --- Print result ---
print(f"{filename} -> {plate_type}")

# --- Save to CSV ---
os.makedirs(os.path.dirname(results_path), exist_ok=True)
df_entry = pd.DataFrame([[filename, plate_type]], columns=["filename", "plate_color"])

# Append if file exists, else create
if os.path.exists(results_path):
    df_entry.to_csv(results_path, mode='a', header=False, index=False)
else:
    df_entry.to_csv(results_path, index=False)
