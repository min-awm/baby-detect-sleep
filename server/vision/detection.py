from PIL import Image, ImageDraw
import io 
import os
from firebase.app import send_firebase
from vision.yolo import get_crib_baby
from vision.face import get_face, is_face_facing_up, compare
from helper.path import get_abs_path

baby_face = get_face(get_abs_path("./data/baby/baby_1.jpg", __file__))[0]
person_face = get_face(get_abs_path("./data/person/person_1.jpeg", __file__))[0]

def draw_boxes(image_path, crib_boxes, baby_boxes, face_boxes=None, output_path="output.jpg"):
    """Vẽ các bounding box lên ảnh và lưu kết quả"""
    # Mở ảnh gốc
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    # Vẽ box cũi (crib) màu xanh
    for box in crib_boxes:
        draw.rectangle(box, outline="green", width=3)
    
    # Vẽ box em bé màu đỏ
    for box in baby_boxes:
        draw.rectangle(box, outline="red", width=3)
    
    # Vẽ box khuôn mặt màu vàng (nếu có)
    if face_boxes:
        for box in face_boxes:
            draw.rectangle(box, outline="yellow", width=2)
    
    # Lưu ảnh kết quả
    image.save(output_path)
    return output_path

def detect(temp):
    CHILD = 0
    CRIB = 1

    crib_babys = get_crib_baby(temp.name)
    crib_baby = crib_babys[0]
        
    crib_boxes = []
    baby_boxes = []
    face_boxes = []

    # Phân loại các bounding box
    for box in crib_baby.boxes:
        cls_id = int(box.cls.item())
        coords = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
        
        if cls_id == CRIB:
            crib_boxes.append(coords)
        elif cls_id == CHILD:
            baby_boxes.append(coords)

    # Hàm tính IoU giữa hai box
    def compute_iou(box1, box2):
        x1, y1, x2, y2 = box1
        x1g, y1g, x2g, y2g = box2

        # Tính phần giao (intersection)
        xi1 = max(x1, x1g)
        yi1 = max(y1, y1g)
        xi2 = min(x2, x2g)
        yi2 = min(y2, y2g)
        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height

        # Tính phần hợp (union)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2g - x1g) * (y2g - y1g)
        union_area = box1_area + box2_area - inter_area

        # Tính IoU
        if union_area == 0:
            return 0
        iou = inter_area / union_area
        return iou

    def is_inside(inner_box, outer_box, iou_threshold=0.1, containment_threshold=0.5):
        """
        Check if inner_box is inside outer_box with more flexible conditions
        Parameters:
            iou_threshold: Minimum IoU to consider overlap
            containment_threshold: Minimum fraction of inner box that must be within outer box
        """
        # Calculate IoU
        iou = compute_iou(inner_box, outer_box)
        
        # Calculate how much of inner box is contained in outer box
        x1, y1, x2, y2 = inner_box
        x1g, y1g, x2g, y2g = outer_box
        
        # Clipped coordinates (intersection)
        xi1 = max(x1, x1g)
        yi1 = max(y1, y1g)
        xi2 = min(x2, x2g)
        yi2 = min(y2, y2g)
        
        # Area calculations
        inner_area = (x2 - x1) * (y2 - y1)
        intersection_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # Either:
        # 1. Significant IoU (original approach), OR
        # 2. Most of the baby box is inside the crib
        return (iou >= iou_threshold) or (intersection_area / inner_area >= containment_threshold)


    # Kiểm tra từng em bé
    baby_in_crib = False
    for baby in baby_boxes:
        for crib in crib_boxes:
            iou = is_inside(baby, crib)
            if iou >= 0.8:
                baby_in_crib = True
                break

    faces = get_face(temp.name)
    for i, face in enumerate(faces):
        landmarks = faces[i]
        face_box = face.bbox.astype(int).tolist()
        face_boxes.append(face_box)  # Thêm box khuôn mặt vào danh sách
        
        if compare(baby_face, face):            
            face_up = is_face_facing_up(landmarks.pose, landmarks.landmark_3d_68)
            if not face_up:
                print("Nam up")
        elif compare(person_face, face):
            print("Nguoi quen")
        else: 
            # Nếu khuôn mặt trong nôi
            in_crib = False
            for crib in crib_boxes:
                iou = is_inside(baby, crib)
                if iou >= 0.8:
                    in_crib = True
                    break

            if in_crib:
                baby_in_crib = True
                face_up = is_face_facing_up(landmarks.pose, landmarks.landmark_3d_68)
                if not face_up:
                    print("Nam up")
                else:
                    print("Nam ngua")
            else:
                print("Nguoi la")

    if not baby_in_crib:
        print("em be khong nam trong noi")
    else:
        print("em be nam trong noi")

    
    send_firebase({
        'baby': {
            'face_up': 'eee',
            'unknown_people': 'ee',
            'no_baby_in_crib': 'w', 
        }
    })
    
    # Vẽ các box lên ảnh
    output_img_path = get_abs_path("./result/output.jpg", __file__)
    
    output_path = draw_boxes(temp.name, crib_boxes, baby_boxes, face_boxes, output_img_path)
    print(f"Đã lưu ảnh kết quả tại: {output_path}")