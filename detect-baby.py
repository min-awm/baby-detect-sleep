import cv2
import mediapipe as mp
import math

# Khởi tạo MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
mp_drawing = mp.solutions.drawing_utils

def vector_2d(point1, point2):
    return (point2.x - point1.x, point2.y - point1.y)

def angle_between_vectors(v1, v2):
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
    if mag1 == 0 or mag2 == 0:
        return 0
    cos_angle = dot / (mag1 * mag2)
    # Giới hạn cos_angle trong khoảng [-1, 1] tránh lỗi math domain error
    cos_angle = max(min(cos_angle, 1), -1)
    angle_rad = math.acos(cos_angle)
    angle_deg = math.degrees(angle_rad)
    return angle_deg

def classify_baby_pose(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        print("❌ Không phát hiện được em bé.")
        return "Không có em bé"

    # Vẽ landmarks lên ảnh
    annotated_image = image.copy()
    mp_drawing.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
        mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
    )

    cv2.imshow("Pose Landmarks", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    keypoints = results.pose_landmarks.landmark
    # Lấy điểm vai trái và hông trái
    left_shoulder = keypoints[mp_pose.PoseLandmark.LEFT_SHOULDER]
    left_hip = keypoints[mp_pose.PoseLandmark.LEFT_HIP]

    # Tính vector thân
    v_body = vector_2d(left_shoulder, left_hip)
    # Vector ngang sang phải
    v_horizontal = (1, 0)

    angle = angle_between_vectors(v_body, v_horizontal)
    print(f"Góc nghiêng thân so với mặt ngang: {angle:.2f}°")

    # Phân loại dựa trên góc nghiêng
    if angle <= 30:
        pose_label = "Nằm ngửa"
    elif angle >= 150:
        pose_label = "Nằm sấp"
    else:
        pose_label = "Nghiêng"

    print(f"✅ Phân loại tư thế: {pose_label}")
    return pose_label

# Thay đường dẫn ảnh phù hợp với bạn
result = classify_baby_pose("./a/nam_sap_18.jpg")
print("Kết quả:", result)
