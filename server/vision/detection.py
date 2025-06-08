import io 
import os
import asyncio
import json
from datetime import datetime
from PIL import Image, ImageDraw
from vision.yolo import get_baby, check_face_in_crib, check_baby_down_pose, get_crib_image
from vision.face import get_faces, get_face_boxes, compare_face
from helper.path import get_abs_path
from firebase.app import send_notification_to_user, get_crib_box
from helper import ws_manager  

baby_face = get_faces(get_abs_path("./data/baby/baby_1.jpg", __file__))[0]
known_face = get_faces(get_abs_path("./data/person/person_1.jpg", __file__))[0]

def draw_boxes(image_path, crib_box, baby_boxes, face_boxes, save_path=False):
    """Vẽ các bounding box lên ảnh và lưu kết quả"""
    output_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = get_abs_path(f"../static/results/{output_name}.jpg", __file__)
    output_image_path = None
    
    # Mở ảnh gốc
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    # Vẽ box cũi (crib) màu xanh
    draw.rectangle(crib_box, outline="green", width=3)
    
    # Vẽ box em bé màu đỏ
    for box in baby_boxes:
        draw.rectangle(box, outline="red", width=3)
    
    for box in face_boxes:
        draw.rectangle(box, outline="yellow", width=2)

    if save_path:
        image.save(output_path)
        output_image_path = f"/admin/results/{output_name}.jpg"


    # Gửi ảnh qua WebSocket
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)
    asyncio.create_task(ws_manager.send_image_client(buffer.read()))
    
    return output_image_path

# Bộ đếm sự kiện
unknown_person_count = 0
baby_not_in_crib_count = 0
baby_down_pose_count = 0
unknown_person_result_global = False
baby_not_in_crib_result_global = False
baby_down_pose_result_global = False

# Ngưỡng xác nhận sự kiện
THRESHOLD = 8
FRAME = 10  

def run_detection(temp):
    """Hàm chạy nhận diện"""
    CHILD = 0
    CRIB = 1

    unknown_person_result = False
    baby_not_in_crib_result = False
    baby_down_pose_result = False

    input_image_path = temp.name
    crib_box=get_crib_box()

    crib_image = get_crib_image(input_image_path, crib_box)
    if not crib_image:
        return "Không có vị trí nôi"

    baby_boxes = get_baby(crib_image)
    if len(baby_boxes) <= 0:
        baby_not_in_crib_result = True

    faces = get_faces(input_image_path)
    face_boxes = get_face_boxes(faces)

    for i, face in enumerate(faces):        
        if check_face_in_crib(crib_box, face_boxes[i]):
            print("Nhận diện được em bé trong nôi")
            baby_not_in_crib_result = False
            continue
        else:
            if compare_face(known_face, face):
                print("Nhận diện được người quen")
                continue
            else:
                print("Nhận diện được người lạ")
                unknown_person_result = True
                continue


    baby_down_pose_result = check_baby_down_pose(crib_image)

    global FRAME, unknown_person_count, baby_not_in_crib_count, baby_down_pose_count, unknown_person_result_global, baby_not_in_crib_result_global, baby_down_pose_result_global
    if unknown_person_result: 
        unknown_person_count += 1

    if baby_not_in_crib_result:
        baby_not_in_crib_count += 1

    if baby_down_pose_count:
        baby_down_pose_count += 1
            
    if FRAME >= 10:
        unknown_person_result_global = unknown_person_count >= THRESHOLD
        baby_not_in_crib_result_global = baby_not_in_crib_count >= THRESHOLD
        baby_down_pose_result_global = baby_down_pose_count >= THRESHOLD
        FRAME = 0
    else:
        FRAME += 1

    print("unknown_person_result", unknown_person_result)
    print("baby_in_crib_result", baby_not_in_crib_result)
    print("baby_down_pose_result", baby_down_pose_result)

    check_send = unknown_person_result_global or unknown_person_result_global or baby_down_pose_result_global
    image_result = draw_boxes(temp.name, crib_box, baby_boxes, face_boxes, save_path=check_send)

    # Vẽ các box lên ảnh    
    if check_send: 
        text_notification = "Cảnh báo:"
        if unknown_person_result_global:
            text_notification += " Có người lạ"
        
        if baby_not_in_crib_result_global:
            text_notification += " Em bé không ở trong nôi"
        
        if baby_down_pose_result_global:
            text_notification += " Em bé nằm úp"
         # send_notification_to_user(text_notification)

    data = {
        "unknown_person_result": unknown_person_result_global,
        "baby_in_crib_result": unknown_person_result_global,
        "baby_down_pose_result": baby_down_pose_result_global,
        "image_result": image_result
    }
    asyncio.create_task(ws_manager.send_text_client(json.dumps(data)))

    return data