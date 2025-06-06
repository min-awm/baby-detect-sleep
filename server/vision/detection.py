import io 
import os
from datetime import datetime
from PIL import Image, ImageDraw
from vision.yolo import get_crib_baby, check_baby_in_crib, check_face_in_crib, check_baby_down_pose, get_crib_image
from vision.face import get_faces, get_face_boxes, compare_face
from helper.path import get_abs_path
from firebase.app import send_notification_to_user

baby_face = get_faces(get_abs_path("./data/baby/baby_1.jpg", __file__))[0]
known_face = get_faces(get_abs_path("./data/person/person_1.jpg", __file__))[0]

def draw_boxes(image_path, crib_boxes, baby_boxes, face_boxes):
    """Vẽ các bounding box lên ảnh và lưu kết quả"""
    output_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = get_abs_path(f"../static/results/{output_name}.jpg", __file__)
    
    # Mở ảnh gốc
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    # Vẽ box cũi (crib) màu xanh
    for box in crib_boxes:
        draw.rectangle(box, outline="green", width=3)
    
    # Vẽ box em bé màu đỏ
    for box in baby_boxes:
        draw.rectangle(box, outline="red", width=3)
    
    for box in face_boxes:
        draw.rectangle(box, outline="yellow", width=2)
    
    # Lưu ảnh kết quả
    image.save(output_path)
    return f"/admin/results/{output_name}.jpg"

def run_detection(temp):
    """Hàm chạy nhận diện"""
    CHILD = 0
    CRIB = 1

    unknown_person_result = False
    baby_in_crib_result = False
    baby_down_pose_result = False

    input_image_path = temp.name

    crib_boxes, baby_boxes = get_crib_baby(input_image_path)
    baby_in_crib_result = check_baby_in_crib(crib_boxes, baby_boxes)

    faces = get_faces(input_image_path)
    face_boxes = get_face_boxes(faces)

    for i, face in enumerate(faces):        
        if compare_face(baby_face, face):            
            baby_in_crib_result = check_face_in_crib(crib_boxes, face_boxes[i])
        elif compare_face(known_face, face):
            # Nếu là người quen, bỏ qua vòng lặp này
            continue
        else: 
            unknown_person_result = True

    crib_image = get_crib_image(input_image_path, crib_boxes)
    if crib_image:
        baby_down_pose_result = check_baby_down_pose(crib_image)

    print("unknown_person_result", unknown_person_result)
    print("baby_in_crib_result", baby_in_crib_result)
    print("baby_down_pose_result", baby_down_pose_result)

    # Vẽ các box lên ảnh    
    image_result = draw_boxes(temp.name, crib_boxes, baby_boxes, face_boxes)

    
    text_notification = "Cảnh báo:"
    if unknown_person_result:
        text_notification += " Có người lạ"
    
    if baby_in_crib_result:
        text_notification += " Em bé không ở trong nôi"
       
    
    if baby_down_pose_result:
        text_notification += " Em bé nằm úp"

    send_notification_to_user(text_notification)

    return {
        "unknown_person_result": unknown_person_result,
        "baby_in_crib_result": baby_in_crib_result,
        "baby_down_pose_result": baby_down_pose_result,
        "image_result": image_result
    }