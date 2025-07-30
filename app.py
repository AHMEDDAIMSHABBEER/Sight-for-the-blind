from flask import Flask, render_template, request, jsonify
import cv2
import os
from ultralytics import YOLO
from gtts import gTTS
from playsound import playsound
import random
import matplotlib.pyplot as plt

# Initialize Flask app
app = Flask(__name__)

# Load the YOLO model
model = YOLO('yolov8n.pt')

# Function to capture image
def capture_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return None

    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None

    image_path = "captured_image.jpg"
    cv2.imwrite(image_path, frame)
    return image_path

# Function to process image
def detect_objects(image_path):
    results = model.predict(image_path, conf=0.5)
    object_counts = {}

    for r in results:
        detected_classes = [r.names[int(c)] for c in r.boxes.cls]
        for obj in detected_classes:
            object_counts[obj] = object_counts.get(obj, 0) + 1

        r = r.plot(conf=False)
        r = cv2.cvtColor(r, cv2.COLOR_BGR2RGB)
        plt.imshow(r)
        plt.savefig("static/detected_image.jpg")
    
    return object_counts

# Function to convert text to speech
def speak_objects(objects):
    if objects:
        detected_str = ", ".join([f"{count} {obj}" for obj, count in objects.items()])
        tts = gTTS(text=f"Detected objects are: {detected_str}", lang='en')
        tts.save("static/detected_objects.mp3")
        return detected_str
    return "No objects detected"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/capture", methods=["POST"])
def capture():
    image_path = capture_image()
    if not image_path:
        return jsonify({"error": "Could not capture image"}), 500
    
    objects = detect_objects(image_path)
    detected_str = speak_objects(objects)
    
    return jsonify({
        "message": detected_str,
        "image_url": "/static/detected_image.jpg",
        "audio_url": "/static/detected_objects.mp3"
    })

if __name__ == "__main__":
    app.run(debug=True)
