import cv2
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
import os
from gtts import gTTS
from playsound import playsound

# Load the pre-trained MobileNet SSD model
print("🔄 Loading model...")
model_url = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1"
model = hub.load(model_url)
print("✅ Model loaded successfully!")

# Class labels (COCO dataset classes)
class_names = [
    "background", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "shoe",
    "eye glasses", "handbag", "tie", "suitcase"
]

# Path to dataset
dataset_path = "Sight-for-the-blind-2/train"  # Update this path if needed

def detect_objects(image):
    """Detect objects in an image using the pre-trained model."""
    h, w, _ = image.shape
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_tensor = tf.convert_to_tensor([img_rgb], dtype=tf.uint8)
    detections = model(input_tensor)

    detected_objects = []

    # Process detections
    for i in range(len(detections["detection_scores"][0])):
        score = detections["detection_scores"][0][i].numpy()
        if score > 0.5:  # Only consider detections above 50% confidence
            class_id = int(detections["detection_classes"][0][i].numpy())
            label = class_names[class_id]
            detected_objects.append(label)

            # Draw bounding box and label
            box = detections["detection_boxes"][0][i].numpy()
            y1, x1, y2, x2 = (box * [h, w, h, w]).astype(int)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image, detected_objects

def speak_objects(objects):
    """Convert detected objects to speech using gTTS."""
    if objects:
        description = "I see " + ", ".join(objects)
        print("🔊 Speaking:", description)

        tts = gTTS(text=description, lang="en")
        audio_file = "object_description.mp3"
        tts.save(audio_file)
        playsound(audio_file)
        os.remove(audio_file)  # Delete the file after playing
    else:
        print("❌ No objects detected.")

def process_dataset():
    """Process all images in the dataset and describe detected objects."""
    if not os.path.exists(dataset_path):
        print("❌ Dataset path not found:", dataset_path)
        return
    
    image_files = [f for f in os.listdir(dataset_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    if not image_files:
        print("❌ No images found in the dataset folder.")
        return

    for image_file in image_files:
        image_path = os.path.join(dataset_path, image_file)
        img = cv2.imread(image_path)

        print(f"🔍 Processing image: {image_file}")
        img, objects = detect_objects(img)

        # Show the image
        cv2.imshow("Detected Objects", img)
        speak_objects(objects)
        cv2.waitKey(2000)  # Show each image for 2 seconds

    cv2.destroyAllWindows()

def process_webcam():
    """Process live webcam feed and describe detected objects."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not access webcam.")
        return

    print("🎥 Webcam started. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Could not read frame.")
            break

        frame, objects = detect_objects(frame)
        speak_objects(objects)

        cv2.imshow("Webcam Object Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()

# Main menu
if __name__ == "__main__":
    print("\n👁️  SIGHT: FOR THE BLIND - OBJECT DETECTION & VOICE ASSISTANCE")
    print("1️⃣ Detect objects from dataset images")
    print("2️⃣ Detect objects in real-time using webcam")
    choice = input("Enter choice (1 or 2): ")

    if choice == "1":
        process_dataset()
    elif choice == "2":
        process_webcam()
    else:
        print("❌ Invalid choice. Exiting...")
