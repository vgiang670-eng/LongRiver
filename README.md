# Hệ Thống AI Giám Sát Hành Vi Bảo Vệ & An Ninh (Guard Behavior Monitoring AI)

Dự án ứng dụng mô hình YOLO để nhận diện hành vi nhân viên bảo vệ và an ninh thông qua camera RTSP thời gian thực.

## 🎯 Các Tính Năng & Hành Vi Nhận Diện
- **Hành vi bất thường:**
  - `phone`: Sử dụng điện thoại trong giờ trực.
  - `smoking`: Hút thuốc lá tại khu vực làm việc.
  - `sleeping`: Ngủ gật, gục đầu trong ca trực.
  - `violence`: Hành vi xô xát, ẩu đả.
- **Rời vị trí trực:** Giám sát vùng ROI (Region of Interest), cảnh báo tự động khi không có nhân sự trực trong vùng quy định quá thời gian cho phép.

---

## 📁 Cấu Trúc Thư Mục
```text
├── DATASETS_GUIDE.md                   # Hướng dẫn chi tiết tổng hợp bộ dữ liệu trên Kaggle
├── KE_HOACH_CHI_TIET_VA_KIEN_TRUC_AI.md # Tài liệu kiến trúc hệ thống và kế hoạch triển khai
├── kaggle_train_yolo.ipynb             # Notebook huấn luyện mô hình YOLOv8 trên Kaggle
├── rtsp_guard_monitor_demo.py          # Script chạy demo giám sát stream RTSP và cảnh báo
├── test_behavior_cam.py                # Script kiểm tra webcam hoặc video mẫu
├── best.pt                             # Trọng số mô hình định dạng PyTorch
├── best.onnx                           # Trọng số mô hình định dạng ONNX (tối ưu hóa suy luận)
├── yolov8n.pt                          # Mô hình nền tảng YOLOv8 Nano
└── guard_behavior_model_weights.zip    # File nén các trọng số đã huấn luyện
```

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy Thử Nghiệm

### 1. Cài đặt môi trường
Yêu cầu Python 3.9+:
```bash
pip install ultralytics opencv-python onnxruntime
```

### 2. Chạy thử nghiệm trên Camera / Webcam
```bash
python test_behavior_cam.py
```

### 3. Chạy giám sát luồng RTSP thực tế
```bash
python rtsp_guard_monitor_demo.py
```

---

## 📦 Quản lý file nặng (Git LFS)
Dự án sử dụng **Git LFS** để lưu trữ các file mô hình có dung lượng lớn (`.onnx`, `.pt`, `.zip`).
Để kéo đầy đủ file trọng số sau khi clone repo:
```bash
git lfs install
git lfs pull
```
