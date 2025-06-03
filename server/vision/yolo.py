from ultralytics import YOLO
import os

script_dir = os.path.dirname(__file__)
rel_path = "./model/yolo.pt"
abs_file_path = os.path.join(script_dir, rel_path)

model = YOLO(abs_file_path)

def get_crib_baby(image):
    results = model(image, save=False)
    return results