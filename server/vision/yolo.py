import os
from PIL import Image
from ultralytics import YOLO
from helper.path import get_abs_path

crib_baby_model = YOLO(get_abs_path("./model/yolo.pt", __file__))
baby_pose_model = YOLO(get_abs_path("./model/baby_pose.pt", __file__))

def get_crib_baby(image):
    """Phát hiện các hộp chứa nôi (crib) và trẻ em (child) trong ảnh"""
    CHILD = 0
    CRIB = 1

    crib_boxes = []
    baby_boxes = []

    crib_babies = crib_baby_model(image, save=False)

    if not crib_babies: 
        return crib_boxes, baby_boxes

    crib_baby = crib_babies[0]

    for box in crib_baby.boxes:
        cls_id = int(box.cls.item())
        boxes = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
        
        if cls_id == CRIB:
            crib_boxes.append(boxes)
        elif cls_id == CHILD:
            baby_boxes.append(boxes)

    return crib_boxes, baby_boxes

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

def check_baby_in_crib(crib_boxes, baby_boxes):
    """Kiểm tra xem có em bé nào nằm trong nôi không"""
    for baby in baby_boxes:
        for crib in crib_boxes:
            if percent_inside(baby, crib) >= 0.8:
                return True

    return False

def check_face_in_crib(crib_boxes, face_box):
    """Kiểm tra xem có mặt em bé nào nằm trong nôi không."""
    for crib in crib_boxes:
        if percent_inside(face_box, crib) >= 0.8:
            return True

    return False

def get_crib_image(input_image_path, crib_boxes):
    """Hàm lấy hình ảnh của nôi em bé"""
    if not crib_boxes:
        return None

    x1 = min(box[0] for box in crib_boxes)
    y1 = min(box[1] for box in crib_boxes)
    x2 = max(box[2] for box in crib_boxes)
    y2 = max(box[3] for box in crib_boxes)

    image = Image.open(input_image_path)
    cropped = image.crop([x1, y1, x2, y2])  
    return cropped

def check_baby_down_pose(image):
    """Kiểm tra xem có em bé có nằm úp không"""
    BABY_LYING_ON_STOMACH = 2
    baby_poses = baby_pose_model(image, save=False)

    if not baby_poses: 
        return False

    baby_pose = baby_poses[0]

    for box in baby_pose[0].boxes:
        class_id = int(box.cls.item())  # 0: 'baby-lying-on-back', 1: 'baby-lying-on-side', 2: 'baby-lying-on-stomach'
        if class_id == BABY_LYING_ON_STOMACH:
            return True

    return False