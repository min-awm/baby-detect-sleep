from ultralytics import YOLO
import os
from helper.path import get_abs_path

model = YOLO(get_abs_path("./model/yolo.pt", __file__))

def get_crib_baby(image):
    results = model(image, save=False)
    return results