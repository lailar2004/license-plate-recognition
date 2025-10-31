import os
import subprocess
import pandas as pd

# --- Helper to run scripts and capture their output ---
def run_script(script_name):
    print(f"\n🚀 Running {script_name} ...\n")
    result = subprocess.run(
        ["python", f"src/{script_name}"],
        capture_output=True,
        text=True,
        encoding='utf-8',  # prevent Unicode errors
        errors='replace'
    )

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print("⚠️ Errors/Warnings:")
        print(result.stderr)
    print(f"\n✅ Completed {script_name}")
    print("-" * 60)
    return result.stdout


# --- Pipeline Execution ---
if __name__ == "__main__":
    print("\n==============================")
    print("📸 LICENSE PLATE RECOGNITION PIPELINE")
    print("==============================\n")

    # 1️⃣ Preprocess and Segment
    run_script("preprocess.py")
    run_script("segment.py")

    # 2️⃣ Get color classifications and OCR outputs
    color_output = run_script("detect_plate_color.py")
    text_output = run_script("recognize_text.py")

    # --- Parse outputs ---
    color_results = {}
    for line in color_output.splitlines():
        if "→" in line or "->" in line:
            parts = line.split("→") if "→" in line else line.split("->")
            if len(parts) == 2:
                filename = parts[0].strip()
                color = parts[1].strip()
                color_results[filename] = color

    text_results = {}
    for line in text_output.splitlines():
        if "→" in line or "->" in line:
            parts = line.split("→") if "→" in line else line.split("->")
            if len(parts) == 2:
                filename = parts[0].strip()
                text = parts[1].strip()
                text_results[filename] = text

    # --- Combine results ---
    combined = []
    for filename in set(list(color_results.keys()) + list(text_results.keys())):
        combined.append({
            "filename": filename,
            "detected_text": text_results.get(filename, "N/A"),
            "plate_color": color_results.get(filename, "N/A")
        })

    # --- Save to CSV ---
    results_dir = "data/results"
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "results.csv")

    df = pd.DataFrame(combined)
    df.to_csv(csv_path, index=False)

    print(f"\n✅ All results saved to: {csv_path}")
    print(df)
    print("\n🎉 All steps completed successfully!")
