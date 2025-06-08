import os
from PIL import Image
from ultralytics import YOLO
from helper.path import get_abs_path

person_model = YOLO("yolo11x.pt")
baby_pose_model = YOLO(get_abs_path("./model/baby_pose.pt", __file__))

def get_baby(image):
    baby = person_model(image, save=False, classes=0)
    baby_boxes = []
    for box in baby[0].boxes:
        boxes = box.xyxy[0].tolist()
        baby_boxes.append(boxes)

    return baby_boxes

def percent_inside(inner_box, outer_box):
    """Tính tỷ lệ diện tích của inner_box nằm trong outer_box"""
    x1, y1, x2, y2 = inner_box
    x1g, y1g, x2g, y2g = outer_box

    inner_area = (x2 - x1) * (y2 - y1)

    # Tọa độ phần giao nhau
    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    # Tỷ lệ phần trăm diện tích của inner_box nằm trong outer_box
    return inter_area / inner_area if inner_area > 0 else 0

def check_face_in_crib(crib_box, face_box):
    """Kiểm tra xem có mặt em bé nào nằm trong nôi không."""
    if percent_inside(face_box, crib_box) >= 0.8:
        return True

    return False

def get_crib_image(input_image_path, crib_box):
    """Hàm lấy hình ảnh của nôi em bé"""
    if not crib_box:
        return None
        
    image = Image.open(input_image_path)
    cropped = image.crop(crib_box)  
    return cropped

def check_baby_down_pose(image):
    """Kiểm tra xem có em bé có nằm úp không"""
    BABY_LYING_ON_STOMACH = 2
    baby_poses = baby_pose_model(image, save=False)

    if not baby_poses: 
        return False

    baby_pose = baby_poses

    for box in baby_pose[0].boxes:
        class_id = int(box.cls.item())  # 0: 'baby-lying-on-back', 1: 'baby-lying-on-side', 2: 'baby-lying-on-stomach'
        if class_id == BABY_LYING_ON_STOMACH:
            return True

    return False