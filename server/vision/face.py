import cv2
import numpy as np
from insightface.app import FaceAnalysis
from numpy.linalg import norm

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=-1, det_size=(640, 640))  # ctx_id=-1 CPU, 0 GPU 

def get_faces(data):
    """Hàm lấy dữ liệu khuôn mặt từ ảnh"""
    img = cv2.imread(data)
    faces = app.get(img)
    return faces

def get_face_boxes(faces):
    """Hàm lấy các hộp khuôn mặt từ ảnh"""
    face_boxes = []
    for i, face in enumerate(faces):
        face_box = face.bbox.astype(int).tolist()
        face_boxes.append(face_box)
    return face_boxes
    
def compare_face(faces1, faces2):   
    """Hàm lấy dữ liệu khuôn mặt từ ảnh""" 
    def cosine_similarity(embedding1, embedding2, eps=1e-5):
        embedding1 = embedding1.ravel()
        embedding2 = embedding2.ravel()
        denominator = norm(embedding1) * norm(embedding2) + eps  # tránh chia cho 0
        return np.dot(embedding1, embedding2) / denominator

    embedding1 = faces1.normed_embedding
    embedding2 = faces2.normed_embedding

    similarity = cosine_similarity(embedding1, embedding2)
    if similarity<0.2:
        return False
    elif similarity>=0.2 and similarity<0.28:
        return True # They are LIKELY TO be the same person
    else:
        return True # 'They ARE the same person'