# README
!pip install ultralytics
from ultralytics import YOLO

# Load a model
model = YOLO("/content/a/runs/detect/train2/weights/last.pt")  # load a pretrained model (recommended for training)
# Train the model
results = model.train(data="/content/a/data.yaml", epochs=500, imgsz=640)
!curl -L "https://universe.roboflow.com/ds/YjAm0zeCKc?key=eQxfcP1oTw" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip
!curl -L "https://universe.roboflow.com/ds/C2HwLyRXLK?key=RSgq45ek0L" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip

from ultralytics import YOLO

# Load model
# model = YOLO("/content/best.pt")
model = YOLO("yolo11m.pt")

# Run detection on video
results = model.predict(source="/content/a.mp4", save=True, save_txt=False, conf=0.25)