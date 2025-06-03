import cv2
import numpy as np
from insightface.app import FaceAnalysis
from numpy.linalg import norm

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=-1, det_size=(640, 640))  # ctx_id=-1 nếu không có GPU

def get_face(data):
    img = cv2.imread(data)
    faces = app.get(img)
    return faces

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
    
def compare(faces1, faces2):    
    # ===== Hàm tính Cosine Similarity =====Add commentMore actions
    def cosine_similarity(embedding1, embedding2, eps=1e-5):
        embedding1 = embedding1.ravel()
        embedding2 = embedding2.ravel()
        denominator = norm(embedding1) * norm(embedding2) + eps  # tránh chia cho 0
        return np.dot(embedding1, embedding2) / denominator

    
    embedding1 = faces1.normed_embedding
    embedding2 = faces2.normed_embedding

    # ===== So sánh hai embedding =====Add commentMore actions
    similarity = cosine_similarity(embedding1, embedding2)

    # ===== Kết luận cơ bản =====
    if similarity<0.2:
        return False
    elif similarity>=0.2 and similarity<0.28:
        return True # They are LIKELY TO be the same person
    else:
        return True # 'They ARE the same person'