# README
!pip install ultralytics
from ultralytics import YOLO

# Load a model
model = YOLO("/content/a/runs/detect/train2/weights/last.pt")  # load a pretrained model (recommended for training)
# Train the model
results = model.train(data="/content/a/data.yaml", epochs=500, imgsz=640)
baby_pose
!curl -L "https://universe.roboflow.com/ds/YjAm0zeCKc?key=eQxfcP1oTw" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip 
!curl -L "https://universe.roboflow.com/ds/C2HwLyRXLK?key=RSgq45ek0L" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip

from ultralytics import YOLO

# Load model
# model = YOLO("/content/best.pt")
model = YOLO("yolo11m.pt")

# Run detection on video
results = model.predict(source="/content/a.mp4", save=True, save_txt=False, conf=0.25)

apt install python3-pip
!pip install gdown
gdown https://drive.google.com/uc?id=1PQ24LH-vpTqp_JFBKi6R5NCLpjOvvtl5
gdown https://drive.google.com/uc?id=1K7jNYLIWXTMceEwiGxqXwNU3FEKisGRI

# Run
git clone http://github.com/min-awm/baby-detect-sleep

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey |sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
&& curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list \
&& sudo apt-get update

sudo apt-get install -y nvidia-container-toolkit

sudo nvidia-ctk runtime configure --runtime=docker

sudo systemctl restart docker