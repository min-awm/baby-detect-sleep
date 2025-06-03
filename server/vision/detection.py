
from PIL import Image
import io 
from vision.yolo import get_crib_baby
from vision.face import check_person

def detect(temp):
    crib_babys = get_crib_baby(temp.name)
    for crib_baby in crib_babys:
        print("----- Kết quả phát hiện -----")
        
        # In tên các lớp (nếu cần)
        print("Các lớp:", crib_baby.names)
        
        # In thông tin bounding boxes
        boxes = crib_baby.boxes
        print("Số vật thể phát hiện:", len(boxes))
        
        for box in boxes:
            print("\n--- Thông tin vật thể ---")
            print("Tọa độ (xyxy):", box.xyxy.tolist())  # [x1, y1, x2, y2]
            print("Độ tin cậy:", box.conf.item())        # Confidence score
            print("ID lớp:", box.cls.item())             # Class ID
            print("Tên lớp:", crib_baby.names[int(box.cls.item())])  # Tên lớp
    # print(crib_baby)

    check = check_person(temp.name)