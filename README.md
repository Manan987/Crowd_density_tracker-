# 🧠 Real-Time Crowd Density Estimation & Alert System

Preventing stampedes and ensuring public safety using AI-powered crowd density monitoring from real-time video streams.

## 📌 Project Overview

This system detects and estimates crowd density using live CCTV video feeds and triggers alerts when predefined thresholds are breached. The goal is to assist in managing crowd flow and preventing dangerous overcrowding at public events, festivals, transportation hubs, and more.

> ⚠️ This implementation **does not use facial recognition or identity tracking**, ensuring privacy compliance. It estimates **density maps** instead.

---

## 🚀 Features

- 🎥 Real-time crowd monitoring via CCTV/RTSP streams
- 🧠 AI-based density map estimation using CNNs (CSRNet, MCNN, SANet)
- 🔔 Automatic alert generation based on critical thresholds
- 💻 Edge computing support (Jetson Nano, Google Coral, etc.)
- 📊 Interactive dashboard with real-time and historical visualization
- 📦 Scalable architecture with modular components (Edge → Server → UI)
- 📩 Multi-channel alerting system (SMS, Email, Slack)

---

## 🧠 Tech Stack

### 📍 Machine Learning
- **Models**: CSRNet, MCNN, SANet
- **Frameworks**: PyTorch (main), TensorFlow (optional)
- **Datasets**: ShanghaiTech, UCF_CC_50, WorldExpo’10, JHU-CROWD++

### 📹 Video Stream Processing
- OpenCV
- FFmpeg / GStreamer (optional)

### ⚙️ Backend
- Python (Flask / FastAPI)
- Node.js with Express (alternative)
- MongoDB / PostgreSQL (data storage)
- Kafka / RabbitMQ (for async alerts)

### 📊 Frontend & Dashboard
- React.js
- Chart.js / D3.js / Plotly
- WebSocket / REST API integration

### 💻 Edge Hardware (Optional)
- NVIDIA Jetson Nano / Orin
- Google Coral USB Accelerator
- Intel OpenVINO Toolkit

---

## 🧱 System Architecture

