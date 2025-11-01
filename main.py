import os
import sys
import subprocess

# --- Define directory paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
RESULTS_DIR = os.path.join(DATA_DIR, "results")

# --- Debug: Print current paths ---
print(f"Base Directory: {BASE_DIR}")
print(f"RAW Directory: {RAW_DIR}")
print(f"Current Working Directory: {os.getcwd()}\n")

# --- Check command-line argument ---
if len(sys.argv) < 2:
    print("Usage: python main.py <image_name>")
    print("\nAvailable files in data/raw:")
    if os.path.exists(RAW_DIR):
        files = os.listdir(RAW_DIR)
        if files:
            for f in files:
                print(f"   - {f}")
        else:
            print("   (No files found)")
    else:
        print(f"    Directory does not exist: {RAW_DIR}")
    sys.exit(1)

filename = sys.argv[1]
raw_path = os.path.join(RAW_DIR, filename)

print(f"Looking for file: {raw_path}")

if not os.path.exists(raw_path):
    print(f"Error: {filename} not found in {RAW_DIR}")
    print("\nAvailable files in data/raw:")
    if os.path.exists(RAW_DIR):
        files = os.listdir(RAW_DIR)
        for f in files:
            print(f"   - {f}")
    sys.exit(1)

print(f"\nStarting License Plate Recognition Pipeline for: {filename}\n")

# --- Helper function to run each stage ---
def run_stage(stage_name, script_filename):
    script_path = os.path.join(SRC_DIR, script_filename)
    print(f"Step: {stage_name}")
    try:
        subprocess.run(
            [sys.executable, script_path, filename],
            check=True,
            cwd=BASE_DIR
        )
        print(f"{stage_name} completed successfully.\n")
    except subprocess.CalledProcessError as e:
        print(f"Error during {stage_name}. Exiting...\n")
        print(e)
        sys.exit(1)

# --- Run all stages in sequence ---
run_stage("Preprocessing Image", "preprocess.py")
run_stage("Detecting Plate with YOLO", "segment.py")
run_stage("Detecting Plate Color", "detect_plate_color.py")
run_stage("Recognizing Text via OCR", "recognize_text.py")

print("All pipeline stages completed successfully!")
print(f"Final results available in: {RESULTS_DIR}")