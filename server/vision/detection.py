import cv2
import numpy as np
from insightface.app import FaceAnalysis
from numpy.linalg import norm

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=-1, det_size=(640, 640))  # ctx_id=-1 nếu không có GPU


def a():
    # ===== Khởi tạo mô hình InsightFace =====
    
    # ===== Hàm tính Cosine Similarity =====
    def cosine_similarity(embedding1, embedding2):
        return np.dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2))

    # ===== Đường dẫn tới 2 ảnh khuôn mặt =====
    img1_path = "/app/vision/a/b2.jpeg"
    img2_path = "/app/vision/a/b3.jpeg"

    # ===== Đọc và phân tích ảnh 1 =====
    img1 = cv2.imread(img1_path)
    faces1 = app.get(img1)
    if not faces1:
        print("Không tìm thấy khuôn mặt trong ảnh 1.")
        exit()

    embedding1 = faces1[0].normed_embedding

    # ===== Đọc và phân tích ảnh 2 =====
    img2 = cv2.imread(img2_path)
    faces2 = app.get(img2)
    if not faces2:
        print("Không tìm thấy khuôn mặt trong ảnh 2.")
        exit()

    embedding2 = faces2[0].normed_embedding

    # ===== So sánh hai embedding =====
    similarity = cosine_similarity(embedding1, embedding2)
    distance = norm(embedding1 - embedding2)

    # ===== In kết quả =====
    print("\n========== KẾT QUẢ SO SÁNH ==========")
    print(f"Cosine Similarity  : {similarity:.4f} (→ gần 1 là giống)")
    print(f"Euclidean Distance : {distance:.4f} (→ gần 0 là giống)")

    # ===== Kết luận cơ bản =====
    if similarity > 0.35 and distance < 1.2:
        print("→ Hai ảnh CÓ THỂ là cùng một người.")
    else:
        print("→ Hai ảnh KHÔNG PHẢI là cùng một người.")