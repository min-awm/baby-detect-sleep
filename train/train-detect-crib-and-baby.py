from ultralytics import YOLO

# Load a model
model = YOLO("yolo11x.pt")

# Train the model
results = model.train(data="./data.yaml", epochs=100, imgsz=640)