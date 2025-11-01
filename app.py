import streamlit as st
import os
import subprocess
import sys
import pandas as pd
from PIL import Image

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
RESULTS_DIR = os.path.join(DATA_DIR, "results")

# Ensure directories exist
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Streamlit Page Setup ---
st.set_page_config(page_title="License Plate Recognition", layout="wide")

# --- Custom CSS for dark theme + white text ---
st.markdown("""
    <style>
        body {
            background-color: #0e1117;
            color: white;
        }
        .title {
            text-align: center;
            color: white;
            font-size: 2.4em;
            font-weight: 700;
        }
        .subtitle {
            text-align: center;
            color: #ccc;
            font-size: 1.1em;
            margin-bottom: 25px;
        }
        .result-card {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0px 4px 15px rgba(255, 255, 255, 0.1);
            margin: 25px auto;
            width: 80%;
            color: white;
        }
        .metric-label {
            color: #aaa;
            font-weight: 600;
            font-size: 1.1em;
        }
        .metric-value {
            font-size: 1.8em;
            font-weight: 700;
            color: #1E90FF;
        }
        .section-header {
            text-align: center;
            font-size: 1.5em;
            font-weight: 600;
            color: white;
            margin-top: 40px;
            margin-bottom: 15px;
        }
        .centered {
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .stButton>button {
            background-color: #1E90FF;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.6em 1.2em;
            font-weight: 600;
        }
        .stButton>button:hover {
            background-color: #3ea0ff;
        }
    </style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("<div class='title'>License Plate Recognition System</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload a vehicle image to automatically detect, recognize, and classify its license plate.</div>", unsafe_allow_html=True)

# --- File Uploader ---
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show smaller preview of uploaded image
    image = Image.open(uploaded_file)
    st.markdown("<div class='centered'>", unsafe_allow_html=True)
    st.image(image, caption="Uploaded Image", width=400)
    st.markdown("</div>", unsafe_allow_html=True)

    # Save uploaded file to data/raw/
    filename = uploaded_file.name
    save_path = os.path.join(RAW_DIR, filename)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Image saved sucessfully")

    # --- Run Processing Button ---
    if st.button("Run License Plate Recognition"):
        with st.spinner("Running full pipeline... Please wait..."):
            try:
                process = subprocess.run(
                    [sys.executable, os.path.join(BASE_DIR, "main.py"), filename],
                    capture_output=True,
                    text=True,
                    cwd=BASE_DIR
                )

                if process.returncode == 0:
                    # --- Load results ---
                    text_csv = os.path.join(RESULTS_DIR, "recognize_text.csv")
                    color_csv = os.path.join(RESULTS_DIR, "detect_plate_color.csv")

                    recognize_text, plate_color = "—", "—"

                    if os.path.exists(text_csv):
                        df_text = pd.read_csv(text_csv)
                        row = df_text[df_text['filename'] == filename]
                        recognize_text = row.iloc[-1]['recognize_text'] if not row.empty else df_text.iloc[-1]['recognize_text']

                    if os.path.exists(color_csv):
                        df_color = pd.read_csv(color_csv)
                        row = df_color[df_color['filename'] == filename]
                        plate_color = row.iloc[-1]['plate_color'] if not row.empty else df_color.iloc[-1]['plate_color']

                    # --- Recognition Summary ---
                    st.markdown("<div class='section-header'>Recognition Summary</div>", unsafe_allow_html=True)
                    st.markdown(f"""
                    <div class='result-card'>
                        <div style='display: flex; justify-content: space-around; text-align: center;'>
                            <div>
                                <div class='metric-label'>Recognized Text</div>
                                <div class='metric-value'>{recognize_text}</div>
                            </div>
                            <div>
                                <div class='metric-label'>Plate Color</div>
                                <div class='metric-value'>{plate_color}</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # --- Display segmented color plate if exists ---
                    segmented_color_path = os.path.join(DATA_DIR, "color_classified", filename)
                    if os.path.exists(segmented_color_path):
                        st.markdown("<div class='section-header'>Detected License Plate</div>", unsafe_allow_html=True)
                        st.markdown("<div style='display: flex; justify-content: center; align-items: center;'>", unsafe_allow_html=True)
                        st.image(segmented_color_path, caption="Segmented Color Plate", width=450)
                        st.markdown("</div>", unsafe_allow_html=True)


                    # --- Pipeline Logs ---
                    st.markdown("<div class='section-header'>Pipeline Output Logs</div>", unsafe_allow_html=True)
                    with st.expander("Show detailed output"):
                        st.code(process.stdout, language="bash")

                else:
                    st.error("Pipeline failed. Check logs below.")
                    with st.expander("Error Details"):
                        st.error(process.stderr)

            except Exception as e:
                st.error(f"Error running pipeline: {e}")

else:
    st.info("📷 Please upload a vehicle image to begin.")
