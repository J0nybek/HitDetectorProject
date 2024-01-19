from flask import Flask, render_template, request
import os
from datetime import datetime
from roboflow import Roboflow
import supervision as sv
import cv2
from PIL import Image
import numpy as np



rf = Roboflow(api_key="ewg29sjKTs1XCYgcfepi")
project = rf.workspace().project("car-damage-wpmh2")
model = project.version(1).model

def check(image_name, timestamp):
    result = model.predict(image_name, confidence=40, overlap=30).json()

    labels = [item["class"] for item in result["predictions"]]

    detections = sv.Detections.from_roboflow(result)

    label_annotator = sv.LabelAnnotator()
    bounding_box_annotator = sv.BoxAnnotator()

    image = cv2.imread(image_name)

    annotated_image = bounding_box_annotator.annotate(
        scene=image, detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections, labels=labels)

    image_data = np.array(annotated_image.copy(), dtype=np.uint8)

    image = Image.fromarray(image_data)
    image.save(f"static/processed/processed_{timestamp}.jpg")

    return f"static/processed/processed_{timestamp}.jpg"


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' in request.files:
        image = request.files['image']
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        file_extension = os.path.splitext(image.filename)[1]
        new_filename = f"upload_{timestamp}{file_extension}"
        # Process and save the image as needed
        image.save('static/uploads/' + new_filename)
        processed_image = check('static/uploads/' + new_filename, timestamp)

        return processed_image  # Return the filename to be displayed on the HTML page
    else:
        return 'No image file provided.'


if __name__ == '__main__':
    app.run(debug=True)