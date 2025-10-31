import easyocr
import cv2
import os
import re

reader = easyocr.Reader(['en'], gpu=False)
input_folder = "data/segmented"

def clean_plate_text(text):
    """Clean OCR output to match Indian plate format"""
    # Remove common OCR mistakes and non-alphanumeric
    text = text.upper().replace(' ', '')
    
    # Remove special characters but keep alphanumeric
    text = re.sub(r'[^A-Z0-9]', '', text)
    
    # Fix common OCR mistakes
    # In first 2 positions (state code), digits should be letters
    if len(text) >= 2:
        # Replace common digit-to-letter mistakes at start
        replacements = {'0': 'O', '1': 'I', '6': 'G', '8': 'B', '5': 'S'}
        first_two = ''.join(replacements.get(c, c) if c.isdigit() else c for c in text[:2])
        text = first_two + text[2:]
    
    # In positions 3-4 (district code), letters should be digits
    if len(text) >= 4:
        replacements = {'O': '0', 'I': '1', 'Z': '2', 'S': '5', 'G': '6', 'B': '8'}
        district = ''.join(replacements.get(c, c) if c.isalpha() else c for c in text[2:4])
        text = text[:2] + district + text[4:]
    
    # In positions 5-6 (series), digits should be letters
    if len(text) >= 6:
        replacements = {'0': 'O', '1': 'I', '6': 'G', '8': 'B', '5': 'S', '2': 'Z'}
        series = ''.join(replacements.get(c, c) if c.isdigit() else c for c in text[4:6])
        text = text[:4] + series + text[6:]
    
    # Last 4 should be digits - fix letter-to-digit mistakes
    if len(text) >= 10:
        replacements = {'O': '0', 'I': '1', 'Z': '2', 'S': '5', 'G': '6', 'B': '8'}
        last_four = ''.join(replacements.get(c, c) if c.isalpha() else c for c in text[-4:])
        text = text[:-4] + last_four
    
    # Indian plate pattern: 2 letters + 2 digits + 1-2 letters + 4 digits
    # Example: TN04BK7999 or MH02DN8718
    
    # Try to extract valid plate pattern
    match = re.search(r'([A-Z]{2}\d{2}[A-Z]{1,2}\d{4})', text)
    if match:
        return match.group(1)
    
    # If we have reasonable length, return it
    if 8 <= len(text) <= 12:
        return text
    
    return text  # Return as-is if no pattern matches

for file in os.listdir(input_folder):
    if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    path = os.path.join(input_folder, file)
    img = cv2.imread(path)
    if img is None:
        continue

    # Convert to grayscale first
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to separate text from background
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours to detect the actual plate region (remove borders)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the bounding box of all text (largest contour area)
    if contours:
        # Get all contours that are likely text (not tiny noise)
        valid_contours = [c for c in contours if cv2.contourArea(c) > 50]
        if valid_contours:
            # Get combined bounding box
            x_min = min([cv2.boundingRect(c)[0] for c in valid_contours])
            y_min = min([cv2.boundingRect(c)[1] for c in valid_contours])
            x_max = max([cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in valid_contours])
            y_max = max([cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in valid_contours])
            
            # Crop to text region with small padding
            padding = 5
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(gray.shape[1], x_max + padding)
            y_max = min(gray.shape[0], y_max + padding)
            
            gray = gray[y_min:y_max, x_min:x_max]

    # Resize to improve OCR - minimum width 400px
    height, width = gray.shape[:2]
    if width < 400:
        scale = 400 / width
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    # Add small padding after cropping
    gray = cv2.copyMakeBorder(gray, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=255)
    
    # Simple thresholding works best for license plates
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Try reading with detail to get confidence scores
    results = reader.readtext(thresh, detail=1, paragraph=False)
    
    # Extract text with highest confidence
    if results:
        text = ''.join([r[1] for r in results]).replace(' ', '').upper()
        text = clean_plate_text(text)
    else:
        text = "unreadable"

    if not text or text == "":
        text = "unreadable"

    print(f"{file} -> {text}")