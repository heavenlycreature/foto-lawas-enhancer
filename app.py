from flask import Flask, render_template, request, redirect, url_for, send_file
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename
from matplotlib import pyplot as plt

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
RESULT_FOLDER = 'static/results/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

def adjust_brightness_contrast(img, brightness=0, contrast=0):
    img = np.int16(img)
    img = img * (contrast / 127 + 1) - contrast + brightness
    img = np.clip(img, 0, 255)
    return np.uint8(img)

def process_image(image_path, brightness=0, contrast=0):
    img = cv2.imread(image_path)

    # Adjust brightness and contrast
    adjusted = adjust_brightness_contrast(img, brightness, contrast)
    cv2.imwrite(os.path.join(RESULT_FOLDER, '0_adjusted.jpg'), adjusted)

    # Convert to grayscale
    gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(RESULT_FOLDER, '1_gray.jpg'), gray)

    # Histogram Equalization
    hist_eq = cv2.equalizeHist(gray)
    cv2.imwrite(os.path.join(RESULT_FOLDER, '2_hist_eq.jpg'), hist_eq)

    # Median Filter
    median = cv2.medianBlur(hist_eq, 3)
    cv2.imwrite(os.path.join(RESULT_FOLDER, '3_median.jpg'), median)

    # Edge Detection (Sobel)
    sobelx = cv2.Sobel(median, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(median, cv2.CV_64F, 0, 1, ksize=3)
 
    sobel = np.sqrt(sobelx**2 + sobely**2)
    sobel = np.uint8(cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX))

    cv2.imwrite(os.path.join(RESULT_FOLDER, '4_sobel.jpg'), sobel)



    results = [
    {"filename": "1_gray.jpg", "label": "Grayscale"},
    {"filename": "2_hist_eq.jpg", "label": "Histogram Equalization"},
    {"filename": "3_median.jpg", "label": "Median Filter"},
    {"filename": "4_sobel.jpg", "label": "Sobel Edge Detection"},
    ]
    return results


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        brightness = int(request.form.get('brightness', 0))
        contrast = int(request.form.get('contrast', 0))

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            result_files = process_image(filepath, brightness, contrast)
            return render_template('result.html', filename='0_adjusted.jpg', results=result_files)
    return render_template('index.html')

@app.route('/uploads/<filename>')
def send_uploaded_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

@app.route('/results/<filename>')
def send_result_file(filename):
    return send_file(os.path.join(app.config['RESULT_FOLDER'], filename))

if __name__ == '__main__':
    app.run(debug=True)
