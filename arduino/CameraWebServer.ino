#include "esp_camera.h"
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>

// 📸 Chọn loại board (AI Thinker)
#define CAMERA_MODEL_AI_THINKER
#include "camera_pins.h"

// 🌐 Wi-Fi cấu hình
const char* ssid = "Galaxy A122FF8";
const char* password = "123456789";

// 🖥️ Server Flask cấu hình
const char* serverIP = "103.70.12.251";
const int serverPort = 8000;

// 🎤 Microphone
#define MIC_PIN 14
#define SOUND_THRESHOLD 1500
#define AUDIO_BUFFER_SIZE 1024
uint16_t audioBuffer[AUDIO_BUFFER_SIZE];
bool isRecording = false;

void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(false);

  // Khởi động camera
  initCamera();

  // Kết nối Wi-Fi
  connectWiFi();

  // Cấu hình microphone
  pinMode(MIC_PIN, INPUT);
  analogSetAttenuation(ADC_11db);

  Serial.println("Start");
}

void loop() {
  int micValue = analogRead(MIC_PIN);
    if (micValue > SOUND_THRESHOLD) {
      Serial.println("Large sound");
      recordAndSendAudio();
    }
  capture_photo();
 
  delay(1000);
}

void initCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0       = Y2_GPIO_NUM;
  config.pin_d1       = Y3_GPIO_NUM;
  config.pin_d2       = Y4_GPIO_NUM;
  config.pin_d3       = Y5_GPIO_NUM;
  config.pin_d4       = Y6_GPIO_NUM;
  config.pin_d5       = Y7_GPIO_NUM;
  config.pin_d6       = Y8_GPIO_NUM;
  config.pin_d7       = Y9_GPIO_NUM;
  config.pin_xclk     = XCLK_GPIO_NUM;
  config.pin_pclk     = PCLK_GPIO_NUM;
  config.pin_vsync    = VSYNC_GPIO_NUM;
  config.pin_href     = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn     = PWDN_GPIO_NUM;
  config.pin_reset    = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size   = FRAMESIZE_QVGA;
  config.jpeg_quality = 12;
  config.fb_count     = 1;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed: 0x%x\n", err);
  } else {
    Serial.println("Camera start");
  }
}

void connectWiFi() {
  WiFi.begin(ssid, password);
  Serial.print("Connect Wi-Fi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nConnect Wi-Fi");
  Serial.print("ESP32 IP: ");
  Serial.println(WiFi.localIP());
}

void capture_photo() {
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Capture fail");
    return;
  }

  Serial.printf("Capture, size: %d bytes\n", fb->len);
  sendPhotoToServer(fb->buf, fb->len);
  esp_camera_fb_return(fb);
}

void sendPhotoToServer(uint8_t* imageData, size_t imageSize) {
  if (WiFi.status() != WL_CONNECTED) return;

  HTTPClient http;
  http.begin("http://" + String(serverIP) + ":" + String(serverPort) + "/upload");
  http.addHeader("Content-Type", "image/jpeg");

  Serial.println("📤 Upload server...");
  int code = http.POST(imageData, imageSize);

  if (code == 200) {
    Serial.println("📡 Send image successfully");
  } else {
    Serial.printf("❌ Error send: %d\n", code);
  }

  http.end();
}

void recordAndSendAudio() {
  Serial.println("🎵 Start recording...");
  isRecording = true;

  unsigned long startTime = millis();
  int index = 0;

  while (millis() - startTime < 2000 && index < AUDIO_BUFFER_SIZE) {
    audioBuffer[index++] = analogRead(MIC_PIN);
    delayMicroseconds(125);
  }

  isRecording = false;
  Serial.printf("🎵 Record sound (%d samples)\n", index);

  size_t audioSize = index * 2;
  uint8_t* audioData = (uint8_t*)audioBuffer;

  sendAudioToServer(audioData, audioSize);
}

void sendAudioToServer(uint8_t* audioData, size_t audioSize) {
  if (WiFi.status() != WL_CONNECTED) return;

  HTTPClient http;
  http.begin("http://" + String(serverIP) + ":" + String(serverPort) + "/sound");
  http.addHeader("Content-Type", "audio/wav");
  http.setTimeout(10000);

  Serial.println("📤 Send sound to server...");
  int code = http.POST(audioData, audioSize);

  if (code == 200) {
    Serial.println("📡 Send sound successfully");
  } else {
    Serial.printf("❌ Error send: %d\n", code);
  }

  http.end();
}
