import cv2
import torch
import time
import numpy as np
import mediapipe as mp
from datetime import datetime
import os
import shutil
import warnings
from collections import deque
import math

from scipy.spatial.distance import cosine
import threading
import queue
import json

# Import các module tự tạo
from ArcFace import ArcFace
from stream import connect_to_stream
from drive.google_drive import upload_to_drive
from config import ESP32_STREAM_URL, GOOGLE_DRIVE_FOLDER_ID

warnings.filterwarnings("ignore", category=FutureWarning)

# Thời gian bắt đầu hệ thống
start_time = time.time()


# ============ SYSTEM CONFIGURATION ============
class Config:
    # Cấu hình xử lý frame
    FRAME_SKIP = 2
    POSE_CONFIDENCE_THRESHOLD = 0.7
    BABY_SIZE_THRESHOLD = 0.15
    POSE_HISTORY_SIZE = 7
    FACE_PROCESS_INTERVAL = 10  # Xử lý khuôn mặt mỗi X frame

    # Ngưỡng phát hiện người lạ (cosine similarity)
    # Giá trị cao hơn = yêu cầu khuôn mặt phải giống hơn để không bị coi là người lạ
    STRANGER_SIMILARITY_THRESHOLD = 0.6

    # Ngưỡng góc tư thế
    TUMMY_TIME_ANGLE_THRESHOLD = 15
    INVERTED_THRESHOLD = 160  # Cho tư thế úp mặt nguy hiểm

    # Khoảng thời gian cảnh báo (giây)
    ALERT_INTERVAL = 30

    # Thư mục lưu trữ
    SNAPSHOTS_DIR = "snapshots"
    KNOWN_FACES_DIR = "known_faces"
    LOGS_DIR = "logs"


config = Config()


# ============ KNOWN FACES LOADER ============
def load_known_faces():
    """Tải các khuôn mặt đã biết sử dụng ArcFace embeddings."""
    known_encodings = []
    known_names = []

    if not os.path.exists(config.KNOWN_FACES_DIR):
        print(f"⚠️ Thư mục {config.KNOWN_FACES_DIR} không tồn tại")
        return known_encodings, known_names

    # Khởi tạo ArcFace model để extract embeddings
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    arcface_temp = ArcFace(device=device)

    if arcface_temp.app is None:
        print("❌ Không thể khởi tạo ArcFace để load faces")
        return known_encodings, known_names

    for filename in os.listdir(config.KNOWN_FACES_DIR):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            name = os.path.splitext(filename)[0]
            filepath = os.path.join(config.KNOWN_FACES_DIR, filename)

            try:
                # Đọc ảnh
                image = cv2.imread(filepath)
                if image is None:
                    print(f"⚠️ Không thể đọc ảnh: {filepath}")
                    continue

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Extract embedding sử dụng ArcFace
                embedding = arcface_temp.get_face_embedding(image_rgb)

                if embedding is not None:
                    known_encodings.append(embedding)
                    known_names.append(name)
                    print(f"✅ Đã tải khuôn mặt: {name}")
                else:
                    print(f"⚠️ Không thể trích xuất embedding cho: {name}")

            except Exception as e:
                print(f"❌ Lỗi xử lý ảnh {filename}: {e}")

    print(f"✅ Đã tải {len(known_encodings)} khuôn mặt đã biết")
    return known_encodings, known_names


def recognize_face_arcface(face_image, known_encodings, known_names, arcface_model, threshold=0.6):
    """
    Nhận diện khuôn mặt sử dụng ArcFace embeddings và cosine similarity.

    Args:
        face_image: Ảnh khuôn mặt (RGB format)
        known_encodings: List các embeddings đã biết
        known_names: List tên tương ứng
        arcface_model: ArcFace model instance
        threshold: Ngưỡng cosine similarity

    Returns:
        (name, confidence) hoặc (None, 0) nếu không nhận diện được
    """
    if not known_encodings or arcface_model is None or arcface_model.app is None:
        return None, 0

    try:
        # Extract embedding từ khuôn mặt input
        current_embedding = arcface_model.get_face_embedding(face_image)

        if current_embedding is None:
            return None, 0

        best_match_idx = None
        best_similarity = 0

        # So sánh với tất cả embeddings đã biết
        for i, known_embedding in enumerate(known_encodings):
            # Tính cosine similarity (1 - cosine distance)
            similarity = 1 - cosine(current_embedding, known_embedding)

            if similarity > best_similarity and similarity > threshold:
                best_similarity = similarity
                best_match_idx = i

        if best_match_idx is not None:
            return known_names[best_match_idx], best_similarity
        else:
            return None, 0

    except Exception as e:
        print(f"❌ Lỗi nhận diện khuôn mặt: {e}")
        return None, 0


# ============ FACE DETECTION CLASS ============
class FaceDetector:
    """Phát hiện khuôn mặt sử dụng OpenCV DNN hoặc Haar Cascade."""

    def __init__(self, confidence_threshold=0.5):
        self.confidence_threshold = confidence_threshold
        self.use_dnn = False

        # Thử sử dụng DNN detector trước
        try:
            # Đường dẫn đến DNN models - đảm bảo các file này có mặt
            dnn_model = 'opencv_face_detector_uint8.pb'
            dnn_config = 'opencv_face_detector.pbtxt'

            if not os.path.exists(dnn_model) or not os.path.exists(dnn_config):
                print(f"⚠️ Không tìm thấy file DNN face detector: {dnn_model}, {dnn_config}")
                raise FileNotFoundError

            self.net = cv2.dnn.readNetFromTensorflow(dnn_model, dnn_config)
            self.use_dnn = True
            print("✅ Sử dụng DNN face detector")
        except Exception as e:
            print(f"⚠️ DNN face detector thất bại: {e}")
            # Fallback về Haar cascade
            haar_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if not os.path.exists(haar_cascade_path):
                print(f"❌ Không tìm thấy file Haar cascade: {haar_cascade_path}")
                print("Vui lòng đảm bảo OpenCV được cài đặt đầy đủ.")
                self.face_cascade = None
            else:
                self.face_cascade = cv2.CascadeClassifier(haar_cascade_path)
            self.use_dnn = False
            print("✅ Sử dụng Haar cascade face detector")

    def detect_faces(self, image):
        """Phát hiện khuôn mặt trong ảnh."""
        if self.use_dnn and hasattr(self, 'net'):
            return self._detect_faces_dnn(image)
        elif not self.use_dnn and hasattr(self, 'face_cascade') and self.face_cascade is not None:
            return self._detect_faces_haar(image)
        else:
            return []  # Không có detector được khởi tạo

    def _detect_faces_dnn(self, image):
        """Phát hiện khuôn mặt sử dụng DNN."""
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
        self.net.setInput(blob)
        detections = self.net.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.confidence_threshold:
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)
                faces.append((x1, y1, x2 - x1, y2 - y1))
        return faces

    def _detect_faces_haar(self, image):
        """Phát hiện khuôn mặt sử dụng Haar cascade."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        return faces.tolist() if len(faces) > 0 else []


# ============ YOLO MODEL LOADER ============
class YOLODetector:
    """Class phát hiện người sử dụng YOLO."""

    def __init__(self):
        self.model = None
        self.load_model()

    def clear_torch_cache(self):
        """Xóa torch cache để tránh xung đột."""
        try:
            cache_dir = torch.hub.get_dir()
            if os.path.exists(cache_dir):
                for item in os.listdir(cache_dir):
                    item_path = os.path.join(cache_dir, item)
                    if 'yolov5' in item.lower():
                        if os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                        elif os.path.isfile(item_path):
                            os.remove(item_path)
            print("✅ Đã xóa torch hub cache")
        except Exception as e:
            print(f"⚠️ Không thể xóa cache: {e}")

    def load_model(self):
        """Tải YOLO model."""
        print("🔄 Đang tải YOLO model tối ưu...")
        try:
            from ultralytics import YOLO
            # Đảm bảo yolov5s.pt được tải về hoặc có sẵn locally
            model = YOLO('yolov5s.pt')
            model.classes = [0]  # Chỉ class 'person'

            class OptimizedYOLOWrapper:
                def __init__(self, model_instance):
                    self.model = model_instance
                    self.classes = [0]

                def __call__(self, frame):
                    # Đặt verbose=False để tắt output trong quá trình inference
                    results = self.model(
                        frame,
                        classes=self.classes,
                        verbose=False,
                        imgsz=416,  # Hoặc 640, điều chỉnh để cân bằng hiệu suất/độ chính xác
                        conf=0.4,
                        iou=0.5
                    )

                    detections = []
                    # ultralytics YOLO trả về danh sách Results objects
                    for r in results:
                        boxes = r.boxes  # Boxes object chứa dữ liệu bounding box
                        if boxes is not None:
                            # xyxy cho (x1, y1, x2, y2)
                            # conf cho confidence score
                            # cls cho class id
                            for i in range(len(boxes)):
                                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                                conf = boxes.conf[i].cpu().numpy()
                                cls = boxes.cls[i].cpu().numpy()
                                detections.append([x1, y1, x2, y2, conf, cls])

                    # Mô phỏng format results cũ của YOLOv5 để tương thích
                    class MockResults:
                        def __init__(self, detections_list):
                            # Chuyển đổi thành numpy array nếu detections_list không rỗng
                            self.xyxy = [np.array(detections_list) if detections_list else np.empty((0, 6))]

                    return MockResults(detections)

            self.model = OptimizedYOLOWrapper(model)
            print("✅ YOLO model được tải thành công")

        except Exception as e:
            print(f"❌ Không thể tải YOLO model: {e}")
            self.model = None


# ============ BABY POSE CLASSIFIER CLASS ============
class EnhancedBabyPoseClassifier:
    """Phân loại tư thế em bé nâng cao."""

    def __init__(self):
        self.pose_history = deque(maxlen=config.POSE_HISTORY_SIZE)
        self.stable_pose = "khong_xac_dinh"
        self.confidence_threshold = 0.6
        self.angle_history = deque(maxlen=5)

    def calculate_angle(self, p1, p2, p3):
        """Tính góc giữa 3 điểm."""
        try:
            v1 = np.array([p1.x - p2.x, p1.y - p2.y])
            v2 = np.array([p3.x - p2.x, p3.y - p2.y])

            cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
            angle = np.arccos(cosine_angle)

            return np.degrees(angle)
        except:
            return 0

    def calculate_body_orientation(self, landmarks):
        """Tính hướng cơ thể."""
        try:
            # Lấy các điểm landmark quan trọng
            nose = landmarks[mp.solutions.pose.PoseLandmark.NOSE.value]
            left_shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]
            left_hip = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value]
            left_ear = landmarks[mp.solutions.pose.PoseLandmark.LEFT_EAR.value]
            right_ear = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_EAR.value]
            left_eye = landmarks[mp.solutions.pose.PoseLandmark.LEFT_EYE.value]
            right_eye = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_EYE.value]

            # Tính các điểm trung tâm
            shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
            shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
            hip_center_x = (left_hip.x + right_hip.x) / 2
            hip_center_y = (left_hip.y + right_hip.y) / 2
            ear_center_y = (left_ear.y + right_ear.y) / 2
            eye_center_y = (left_eye.y + right_eye.y) / 2

            # Tính vector cơ thể (từ vai đến hông)
            body_vector_x = hip_center_x - shoulder_center_x
            body_vector_y = hip_center_y - shoulder_center_y
            body_angle = math.atan2(body_vector_y, body_vector_x)
            body_angle_degrees = abs(math.degrees(body_angle))

            # Các tham số phân tích khác
            nose_to_ear_y = nose.y - ear_center_y
            nose_to_eye_y = nose.y - eye_center_y
            shoulder_diff = abs(left_shoulder.y - right_shoulder.y)  # Chênh lệch dọc giữa các vai

            return {
                'body_angle_degrees': body_angle_degrees,
                'nose_to_ear_y': nose_to_ear_y,
                'nose_to_eye_y': nose_to_eye_y,
                'shoulder_diff': shoulder_diff,
                'shoulder_center_y': shoulder_center_y,
                'hip_center_y': hip_center_y,
                'nose_y': nose.y,
                'body_vertical': abs(body_angle_degrees - 90) < 30,  # Góc gần 90 độ (dọc)
                'body_horizontal': body_angle_degrees < 30 or body_angle_degrees > 150  # Góc gần 0 hoặc 180 (ngang)
            }
        except Exception as e:
            return None

    def classify_pose_enhanced(self, landmarks, image_width, image_height):
        """Phân loại tư thế nâng cao."""
        orientation = self.calculate_body_orientation(landmarks)
        if not orientation:
            return "khong_xac_dinh", 0.0

        confidence = 0.0
        pose_label = "khong_xac_dinh"

        # Lưu trữ lịch sử góc để ổn định
        self.angle_history.append(orientation['body_angle_degrees'])
        avg_angle = sum(self.angle_history) / len(self.angle_history) if self.angle_history else 0

        # Phân loại tư thế
        # Nằm ngửa (On back)
        if (orientation['nose_to_ear_y'] > 0.07 and  # Mũi cao hơn đường tai đáng kể
                orientation['shoulder_diff'] > 0.03 and  # Vai tương đối ngang
                orientation['body_horizontal'] and  # Cơ thể nằm ngang
                abs(avg_angle) < 10):  # Góc cơ thể gần ngang
            pose_label = "nam_ngua"
            confidence = min(0.95, 0.75 + abs(orientation['nose_to_ear_y']) * 5)

        # Tummy Time (Nằm sấp, đầu nâng lên)
        elif (orientation['nose_to_ear_y'] < 0.05 and orientation[
            'nose_to_ear_y'] > -0.05 and  # Mũi xung quanh đường tai
              orientation['nose_to_eye_y'] > 0.03 and  # Mũi dưới đường mắt (đầu ngẩng lên)
              orientation['body_horizontal'] and  # Cơ thể nằm ngang
              orientation['shoulder_center_y'] < orientation['hip_center_y'] and  # Vai cao hơn hông
              abs(avg_angle) < 15):  # Góc cơ thể gần ngang
            pose_label = "tummy_time"
            confidence = min(0.9, 0.7 + abs(orientation['nose_to_ear_y'] - 0.005) * 10)

        # Nằm sấp (On stomach, úp mặt)
        elif (orientation['nose_to_ear_y'] < -0.05 and  # Mũi thấp hơn đường tai đáng kể
              orientation['body_horizontal'] and  # Cơ thể nằm ngang
              orientation['shoulder_diff'] < 0.1 and  # Vai tương đối nằm ngang
              abs(avg_angle) < 10):  # Góc cơ thể gần ngang
            pose_label = "nam_sap"
            confidence = min(0.95, 0.65 + abs(orientation['nose_to_ear_y']) * 8)

        # Nằm úp 180 (Inverted/Nguy hiểm - đầu hoàn toàn hướng xuống)
        elif (orientation['body_horizontal'] and  # Cơ thể nằm ngang
              orientation['hip_center_y'] < orientation['shoulder_center_y'] and  # Hông cao hơn vai (úp ngược)
              abs(avg_angle) > 170):  # Góc cơ thể gần 180 độ
            pose_label = "nam_up_180"  # nguy hiểm
            confidence = min(0.95, 0.7 + abs(orientation['nose_to_ear_y']) * 8)

        return pose_label, confidence

    def update_pose_history(self, pose_label, confidence):
        """Cập nhật lịch sử tư thế."""
        if confidence > self.confidence_threshold:
            self.pose_history.append((pose_label, confidence, time.time()))

        # Xóa dữ liệu cũ
        current_time = time.time()
        self.pose_history = deque(
            [(pose, conf, t) for pose, conf, t in self.pose_history if current_time - t < 3.0],  # Giữ 3 giây gần nhất
            maxlen=config.POSE_HISTORY_SIZE
        )

        # Xác định tư thế ổn định
        if len(self.pose_history) >= 3:  # Cần ít nhất 3 entry gần đây để ổn định
            recent_poses = [(p[0], p[1]) for p in list(self.pose_history)[-3:]]
            pose_counts = {}

            for pose, conf in recent_poses:
                if pose not in pose_counts:
                    pose_counts[pose] = []
                pose_counts[pose].append(conf)

            best_pose = None
            best_score = 0

            for pose, confidences in pose_counts.items():
                avg_conf = sum(confidences) / len(confidences)
                # Score tính cả confidence trung bình và tần suất
                score = avg_conf * (len(confidences) / len(recent_poses))

                if score > best_score:
                    best_score = score
                    best_pose = pose

            if best_pose and best_score > 0.4:  # Score hợp lý để coi là ổn định
                self.stable_pose = best_pose
                return self.stable_pose, best_score

        return self.stable_pose, 0.5  # Trả về stable hiện tại hoặc default nếu chưa đủ lịch sử


# ============ UTILITY FUNCTIONS ============
def is_baby_size(box, frame_width, frame_height):
    """Kiểm tra bounding box có trong khoảng kích thước em bé không."""
    x1, y1, x2, y2 = box
    box_width = x2 - x1
    box_height = y2 - y1
    box_area = box_width * box_height
    frame_area = frame_width * frame_height
    relative_size = box_area / frame_area
    # Điều chỉnh ngưỡng này dựa trên kích thước em bé thông thường so với góc nhìn camera
    return 0.01 <= relative_size <= 0.25  # Giảm kích thước tối đa để phù hợp với em bé


def is_child_face(face_location, baby_boxes, margin=20):
    """Kiểm tra khuôn mặt có phải của em bé không."""
    x, y, w, h = face_location
    face_center_x = x + w // 2
    face_center_y = y + h // 2

    for bx1, by1, bx2, by2 in baby_boxes:
        # Kiểm tra tâm khuôn mặt có nằm trong bounding box em bé với margin không
        if (bx1 - margin <= face_center_x <= bx2 + margin and
                by1 - margin <= face_center_y <= by2 + margin):
            return True
    return False


def save_alert_snapshot(frame, alert_type, timestamp):
    """Lưu snapshot cảnh báo."""
    try:
        os.makedirs(config.SNAPSHOTS_DIR, exist_ok=True)
        filename = f"{alert_type}_{int(timestamp)}.jpg"
        filepath = os.path.join(config.SNAPSHOTS_DIR, filename)
        cv2.imwrite(filepath, frame)
        print(f"📸 Đã lưu snapshot: {filepath}")
        return filepath
    except Exception as e:
        print(f"❌ Lỗi lưu snapshot: {e}")
        return None


def log_event(event_type, message, timestamp=None):
    """Ghi log sự kiện hệ thống."""
    try:
        os.makedirs(config.LOGS_DIR, exist_ok=True)
        if timestamp is None:
            timestamp = time.time()

        log_entry = {
            'timestamp': timestamp,
            'datetime': datetime.fromtimestamp(timestamp).isoformat(),
            'event_type': event_type,
            'message': message
        }

        log_file = os.path.join(config.LOGS_DIR, f"events_{datetime.now().strftime('%Y%m%d')}.json")

        logs = []
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    logs = json.load(f)
            except json.JSONDecodeError:
                logs = []  # Nếu file bị lỗi, bắt đầu mới

        logs.append(log_entry)

        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)

    except Exception as e:
        print(f"❌ Lỗi ghi log: {e}")


# ============ MAIN MONITORING SYSTEM CLASS ============
class BabyMonitorSystem:
    """Hệ thống giám sát em bé chính."""

    def __init__(self):
        self.initialize_components()
        self.reset_counters()

    def initialize_components(self):
        """Khởi tạo các thành phần hệ thống."""
        print("🚀 Đang khởi tạo hệ thống giám sát em bé thông minh...")

        # MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=config.POSE_CONFIDENCE_THRESHOLD,
            min_tracking_confidence=0.5
        )

        # Detectors
        self.face_detector = FaceDetector(confidence_threshold=0.7)
        self.yolo_detector = YOLODetector()
        self.baby_classifier = EnhancedBabyPoseClassifier()
        # Khởi tạo ArcFace model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.arcface_model = ArcFace(device=device)
        if self.arcface_model is not None:
            print(f"✅ ArcFace model được tải thành công trên {device}")
        else:
            print("❌ Khởi tạo ArcFace model thất bại. Tắt nhận diện khuôn mặt.")
            self.arcface_model = None

        # Tải các khuôn mặt đã biết
        self.known_encodings, self.known_names = load_known_faces()

        # Kết nối video stream
        self.cap = connect_to_stream(ESP32_STREAM_URL)
        if not self.cap or not self.cap.isOpened():
            print(f"❌ Kết nối stream thất bại tại {ESP32_STREAM_URL}. Thoát.")
            exit()  # Thoát nếu kết nối stream thất bại

        print("✅ Hệ thống sẵn sàng!")

    def reset_counters(self):
        """Reset counters và biến theo dõi."""
        self.frame_count = 0
        self.last_alert_time = {
            'face': 0,
            'baby_dangerous_pose': 0,
            'baby_tummy_time': 0,
            'no_baby': 0,
        }
        self.baby_detected_in_frame = False

    def handle_stranger_alert(self, frame, current_time):
        """Kích hoạt hành động cho cảnh báo người lạ."""
        if current_time - self.last_alert_time['face'] > config.ALERT_INTERVAL:
            print(f"🚨 CẢNH BÁO: Người lạ phát hiện! ({datetime.now().strftime('%H:%M:%S')})")
            log_event("stranger_detected", "Người lạ phát hiện.")
            self.last_alert_time['face'] = current_time

            snapshot_path = save_alert_snapshot(frame, "stranger_alert", current_time)
            if snapshot_path:
                # Sử dụng threading cho upload để tránh blocking main loop
                threading.Thread(target=upload_to_drive, args=(snapshot_path, GOOGLE_DRIVE_FOLDER_ID)).start()
                print(f"📤 Đang upload snapshot lên Google Drive: {snapshot_path}")

    def process_face_recognition(self, frame, faces, baby_boxes, current_time):
        """Xử lý nhận diện khuôn mặt sử dụng ArcFace embeddings và cosine similarity."""
        for (x, y, w, h) in faces:
            # Bỏ qua khuôn mặt có thể là của em bé
            if is_child_face((x, y, w, h), baby_boxes):
                continue

            # Trích xuất vùng khuôn mặt với padding
            padding = 10
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(frame.shape[1], x + w + padding)
            y2 = min(frame.shape[0], y + h + padding)

            face_img_cropped_bgr = frame[y1:y2, x1:x2]
            if face_img_cropped_bgr.size == 0:
                continue

            face_img_cropped_rgb = cv2.cvtColor(face_img_cropped_bgr, cv2.COLOR_BGR2RGB)

            # Nhận diện khuôn mặt
            name, confidence = recognize_face_arcface(
                face_img_cropped_rgb,
                self.known_encodings,
                self.known_names,
                self.arcface_model,
                config.STRANGER_SIMILARITY_THRESHOLD
            )

            # Xử lý kết quả nhận diện
            label = "Stranger"
            color = (0, 0, 255)  # Red for stranger

            if name:
                label = f"{name} ({confidence:.1%})"
                color = (0, 255, 0)  # Green for known person
            else:
                # Trigger alert for unknown person
                self.handle_stranger_alert(frame, current_time)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def process_frame(self, frame):
        """Processes a single video frame."""
        current_time = time.time()
        height, width = frame.shape[:2]

        # Convert to RGB for MediaPipe and ArcFace
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Tracking variables for current frame
        baby_detected_current = False
        baby_boxes = []
        baby_status = "khong_xac_dinh"

        # Process person detection (for baby)
        if self.yolo_detector.model:
            results = self.yolo_detector.model(frame)  # YOLO works on BGR
            if results.xyxy and results.xyxy[0].ndim == 2 and results.xyxy[0].shape[1] >= 6:
                people_detections = results.xyxy[0]
            else:
                people_detections = []

            for detection in people_detections:
                x1, y1, x2, y2, conf, cls = detection[:6]
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                if not is_baby_size([x1, y1, x2, y2], width, height):
                    continue

                baby_boxes.append([x1, y1, x2, y2])
                baby_detected_current = True

                # Extract baby region for pose estimation
                baby_img = rgb_frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
                if baby_img.size == 0:
                    continue

                # Pose analysis
                pose_results = self.pose.process(baby_img)
                if pose_results.pose_landmarks:
                    pose_label, confidence = self.baby_classifier.classify_pose_enhanced(
                        pose_results.pose_landmarks.landmark, x2 - x1, y2 - y1
                    )

                    stable_pose, stable_confidence = self.baby_classifier.update_pose_history(
                        pose_label, confidence
                    )

                    baby_status = stable_pose

                    # Choose color based on pose
                    color = (0, 255, 0)  # Green - normal
                    if baby_status in ["nam_up_180", "nghiem_trong"]:  # "inverted" -> "nghiem_trong" or similar
                        color = (0, 0, 255)  # Red - dangerous
                    elif baby_status == "tummy_time":
                        color = (255, 255, 0)  # Yellow - tummy time

                    # Draw bounding box and pose status
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        f"Baby: {baby_status} ({stable_confidence:.1%})",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                    )
                else:
                    # If no pose landmarks, still draw the baby box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)  # Orange for detected but no pose
                    cv2.putText(
                        frame,
                        "Baby: Pose N/A",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2
                    )

        # Process face recognition (for strangers)
        if self.arcface_model and self.arcface_model.app and self.frame_count % config.FACE_PROCESS_INTERVAL == 0:
            faces = self.face_detector.detect_faces(frame)

            self.process_face_recognition(frame, faces, baby_boxes, current_time)

        # Check for "no baby detected" alert
        if not baby_detected_current and self.baby_detected_in_frame:  # Baby was detected, but now isn't
            if current_time - self.last_alert_time['no_baby'] > config.ALERT_INTERVAL:
                print(f"🚨 CẢNH BÁO: Không thấy em bé! ({datetime.now().strftime('%H:%M:%S')})")
                log_event("no_baby_detected", "Không thấy em bé.")
                self.last_alert_time['no_baby'] = current_time
                snapshot_path = save_alert_snapshot(frame, "no_baby_alert", current_time)
                if snapshot_path:
                    threading.Thread(target=upload_to_drive, args=(snapshot_path, GOOGLE_DRIVE_FOLDER_ID)).start()

        # Check for dangerous pose alert
        if baby_detected_current:
            if baby_status in ["nam_up_180", "nam_sap"] and current_time - self.last_alert_time[
                'baby_dangerous_pose'] > config.ALERT_INTERVAL:
                print(f"🚨 CẢNH BÁO: Em bé ở tư thế nguy hiểm: {baby_status} ({datetime.now().strftime('%H:%M:%S')})")
                log_event("dangerous_pose", f"Em bé ở tư thế nguy hiểm: {baby_status}.")
                self.last_alert_time['baby_dangerous_pose'] = current_time
                snapshot_path = save_alert_snapshot(frame, "dangerous_pose_alert", current_time)
                if snapshot_path:
                    threading.Thread(target=upload_to_drive, args=(snapshot_path, GOOGLE_DRIVE_FOLDER_ID)).start()

            # Check for tummy time notification
            elif baby_status == "tummy_time" and current_time - self.last_alert_time[
                'baby_tummy_time'] > config.ALERT_INTERVAL:
                print(f"ℹ️ THÔNG BÁO: Em bé đang tummy time ({datetime.now().strftime('%H:%M:%S')})")
                log_event("tummy_time", "Em bé đang tummy time.")
                self.last_alert_time['baby_tummy_time'] = current_time

        self.baby_detected_in_frame = baby_detected_current  # Update status for next frame

        # Display status information
        frame_time_taken = time.time() - current_time  # Time taken to process this frame
        fps = 1.0 / frame_time_taken if frame_time_taken > 0 else 0

        status_lines = [
            f"Baby: {'Yes' if baby_detected_current else 'No'}",
            f"Pose: {baby_status if baby_detected_current else 'N/A'}",
            f"FPS: {fps:.1f}",
            f"Frame: {self.frame_count}",
            f"Time: {datetime.now().strftime('%H:%M:%S')}"
        ]
        for i, text in enumerate(status_lines):
            y_pos = 30 + i * 25
            cv2.putText(frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Baby Monitor", frame)
        self.frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 Shutting down system...")
            return False  # Signal to stop loop
        return True  # Signal to continue loop


# Main execution block
if __name__ == "__main__":
    # Ensure necessary directories exist
    os.makedirs(config.SNAPSHOTS_DIR, exist_ok=True)
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    os.makedirs(config.KNOWN_FACES_DIR, exist_ok=True)  # Ensure known_faces is there

    try:
        baby_monitor = BabyMonitorSystem()
        if not baby_monitor.cap or not baby_monitor.cap.isOpened():
            print("❌ Exiting due to failed stream connection.")
        else:
            while True:
                ret, frame = baby_monitor.cap.read()
                if not ret:
                    print("❌ Could not read frame from stream. Exiting.")
                    break
                if not baby_monitor.process_frame(frame):  # process_frame now returns True/False
                    break  # Break loop if process_frame signals to stop

    except KeyboardInterrupt:
        print("🛑 User stopped the program.")
    except Exception as ex:
        print(f"❌ System error: {ex}")
        import traceback
        traceback.print_exc()
    finally:
        if 'baby_monitor' in locals() and hasattr(baby_monitor, 'cap') and baby_monitor.cap.isOpened():
            baby_monitor.cap.release()
        if 'baby_monitor' in locals() and hasattr(baby_monitor, 'pose') and baby_monitor.pose:
            baby_monitor.pose.close()
        cv2.destroyAllWindows()
        if 'baby_monitor' in locals():
            print(f"✅ Resources released. Total frames processed: {baby_monitor.frame_count}")
