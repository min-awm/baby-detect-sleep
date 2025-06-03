import insightface
import cv2
import numpy as np
import matplotlib.pyplot as plt

def draw_landmark_3d_68(img, faces):
    print(f"Đã tìm thấy {len(faces)} khuôn mặt.")
    for i, face in enumerate(faces):
        print(f"Khuôn mặt {i+1}:")
        print("  - Vector đặc trưng:", face.embedding[:5], "...")

        # Vẽ bounding box
        box = face.bbox.astype(int)
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

        # Vẽ các điểm landmark 3D (dạng (68, 3))
        if hasattr(face, 'landmark_3d_68') and face.landmark_3d_68 is not None:
            landmarks = face.landmark_3d_68
            for (x, y, z) in landmarks:
                cv2.circle(img, (int(x), int(y)), 1, (0, 0, 255), -1)
        else:
            print("  - Không có landmark_3d_68.")

    # Hiển thị kết quả (chuyển BGR -> RGB)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 8))
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.title("Landmark 3D (68 điểm)")
    plt.show()

def is_face_facing_up(pose, landmark_3d_68, threshold_pitch=15):
    """
    Xác định mặt đang úp hay ngửa dựa vào góc pitch (pose) và landmark 3D

    Args:
        pose: mảng [yaw, pitch, roll] (đơn vị độ hoặc radian tùy hệ thống)
        landmark_3d_68: mảng 68 điểm landmark 3D (x,y,z)
        threshold_pitch: ngưỡng phân biệt (độ)

    Returns:
        True nếu mặt ngửa lên
        False nếu mặt úp xuống
    """
    # Lấy pitch, chuyển sang độ nếu cần (ở đây giả sử pitch là radian)
    pitch = pose[1]
    pitch_deg = np.degrees(pitch) if abs(pitch) < 2*np.pi else pitch  # nếu đã là độ thì giữ nguyên

    # Kiểm tra pitch đơn giản
    if pitch_deg > threshold_pitch:
        # Mặt đang ngửa lên
        return True
    elif pitch_deg < -threshold_pitch:
        # Mặt đang cúi xuống (úpside down)
        return False

    # Nếu pitch gần 0, dùng phân tích sâu hơn landmark 3D

    # Lấy điểm mũi (landmark số 30 thường là mũi đầu mũi trong 68 điểm)
    nose_point = landmark_3d_68[30]  # (x,y,z)

    # Lấy điểm mắt trái và mắt phải (thường 36-39 là mắt trái, 42-45 mắt phải)
    left_eye_points = landmark_3d_68[36:42]
    right_eye_points = landmark_3d_68[42:48]

    # Trung bình tọa độ z của mắt trái và mắt phải
    eye_z_avg = (np.mean(left_eye_points[:,2]) + np.mean(right_eye_points[:,2])) / 2.0

    # So sánh z mũi và mắt
    # Nếu mũi nằm sau mắt (z mũi lớn hơn z mắt) nghĩa là mặt cúi xuống
    if nose_point[2] > eye_z_avg + 1.0:  # ngưỡng 1.0 có thể điều chỉnh theo dữ liệu
        return False  # mặt úp
    else:
        return True   # mặt ngửa



def draw_keypoints(img, kps):
    """
    Vẽ các điểm keypoints 2D lên ảnh.
    Args:
        img: ảnh đầu vào (numpy array)
        kps: mảng numpy shape (N, 2) chứa tọa độ x,y của các điểm keypoints
    """
    for i, (x, y) in enumerate(kps):
        cv2.circle(img, (int(x), int(y)), 3, (255, 0, 0), -1)  # điểm màu xanh dương, kích thước 3
        cv2.putText(img, str(i+1), (int(x)+5, int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 8))
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.title("Landmark 3D (68 điểm)")
    plt.show()

def draw_landmark_2d_68(img, faces):
    print(f"Đã tìm thấy {len(faces)} khuôn mặt.")
    for i, face in enumerate(faces):
        print(f"Khuôn mặt {i+1}:")
        print("  - Vector đặc trưng:", face.embedding[:5], "...")

        # Vẽ bounding box
        box = face.bbox.astype(int)
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

        # Vẽ các điểm landmark 2D (dạng (68, 2))
        if hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None:
            landmarks = face.landmark_2d_106  # mảng Nx2
            for (x, y) in landmarks:
                cv2.circle(img, (int(x), int(y)), 2, (255, 0, 0), -1)  # Màu xanh dương
        else:
            print("  - Không có landmark_2d_106.")

    # Hiển thị kết quả (chuyển BGR -> RGB để hiển thị đúng màu)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 8))
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.title("Landmark 2D (68 điểm)")
    plt.show()

# Khởi tạo mô hình
face_model = insightface.app.FaceAnalysis(name='buffalo_s')
face_model.prepare(ctx_id=0)  # hoặc ctx_id=-1 nếu không có GPU

# Đọc ảnh
img_path = "/content/a3.png"  # Thay bằng đường dẫn ảnh của bạn (đã upload)
img = cv2.imread(img_path)

# Phát hiện khuôn mặt
faces = face_model.get(img)
landmarks = faces[0]
# print(landmarks)

if len(faces) == 0:
    print("Không tìm thấy khuôn mặt nào.")
else:
    print(f"Đã tìm thấy {len(faces)} khuôn mặt.")
    # for i, face in enumerate(faces):
    #     print(f"Khuôn mặt {i+1}:")
    #     print("  - Vector đặc trưng:", face.embedding[:5], "...")
    #     box = face.bbox.astype(int)
    #     cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

    # # Chuyển BGR → RGB để hiển thị đúng màu
    # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # plt.figure(figsize=(10, 8))
    # plt.imshow(img_rgb)
    # plt.axis("off")
    # plt.title("Khuôn mặt đã phát hiện")
    # plt.show()
    # draw_landmark_3d_68(img, faces)
    draw_landmark_2d_68(img, faces)
    # draw_keypoints(img, landmarks.kps)
    # print(landmarks.landmark_2d_106[30])
    if is_face_facing_up(landmarks.pose, landmarks.landmark_3d_68):
      print("Face is ngửa lên.")
    else:
      print("Face is up.")

