import os
from flask import Flask, request, jsonify
import easyocr
import cv2
import numpy as np

app = Flask(__name__)
reader = easyocr.Reader(['en'], gpu=False)

@app.route('/ocr', methods=['POST'])
def ocr():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400

        image = request.files['image']
        if image.filename == '':
            return jsonify({"error": "Empty filename"}), 400

        # Read image
        img_bytes = image.read()
        img_np = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        # OCR processing
        results = reader.readtext(img)  # Use original image instead of grayscale

        # Convert coordinates to native Python types
        formatted_results = []
        for (bbox, text, confidence) in results:
            formatted_results.append({
                "text": text,
                "confidence": float(confidence),
                "position": [
                    [int(x) for x in point]  # Convert coordinates to integers
                    for point in bbox
                ]
            })

        return jsonify({"results": formatted_results})

    except Exception as e:
        return jsonify({"error": f"OCR failed: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
