# KẾ HOẠCH & KIẾN TRÚC TRIỂN KHAI MODEL AI PHÂN TÍCH HÀNH VI BẢO VỆ
> **Mục tiêu**: Nghiên cứu, xây dựng và triển khai AI phân tích 5 hành vi: *Ngủ, Sử dụng điện thoại, Hút thuốc, Bỏ vị trí trực, Hành vi bạo lực* từ Camera giám sát.  
> **Thời hạn (Deadline)**: 30/10  
> **Nền tảng huấn luyện**: Kaggle (GPU T4 x2 / P100 miễn phí)  
> **Môi trường triển khai mục tiêu**: RTSP Camera / NVR / Edge AI PC / Mini Server

---

## 1. PHÂN TÍCH BẢN CHẤT 5 HÀNH VI & CHIẾN LƯỢC KỸ THUẬT

Một sai lầm phổ biến khi mới làm bài toán này là cố gắng tìm hoặc train **1 model duy nhất** để nhận diện cả 5 hành vi trên từng frame ảnh tĩnh. Trong thực tế camera giám sát an ninh:

| Hành vi | Bản chất bài toán | Phương pháp AI tối ưu | Độ khó & Lưu ý thực tế |
| :--- | :--- | :--- | :--- |
| **1. Hút thuốc (Smoking)** | Object Detection + Pose/Hand Interaction | **YOLOv8/YOLOv11 Object Detection**: Phát hiện điếu thuốc (`cigarette`), tư thế đưa tay lên miệng (`smoking`). | Điếu thuốc rất nhỏ, camera góc xa khó thấy. Cần phát hiện cử chỉ tay gần mặt + khói hoặc bbox thuốc. |
| **2. Dùng điện thoại (Phone)** | Object Detection + Interaction | **YOLOv8/YOLOv11 Object Detection**: Phát hiện điện thoại (`cellphone`), người cầm điện thoại (`holding_phone`), nghe điện thoại (`calling`). | Dataset có sẵn rất nhiều trên Roboflow / Kaggle. |
| **3. Ngủ khi trực (Sleeping)** | Posture + Temporal (thời gian) | **YOLO-Pose + Motion/Head tilt**: Phát hiện đầu gục xuống bàn, tư thế ngả lưng nhắm mắt, và **không cử động trong $T > 30-60$ giây**. | Không nên chỉ bắt frame ảnh tĩnh vì bảo vệ có thể chỉ đang cúi đầu ghi sổ sách 2 giây. Cần bộ lọc thời gian (Temporal Smoothing). |
| **4. Bỏ vị trí trực (Leaving post)** | Zone Detection + Tracking (Không cần train thêm model) | **YOLO (Person Detection) + ByteTrack + ROI Polygon**: Vẽ vùng chốt bảo vệ (ROI). Nếu số lượng `person` trong vùng = 0 trong thời gian $T > X$ phút $\rightarrow$ Báo động. | Giải pháp chuẩn xác 100%, cực nhẹ, không tốn tài nguyên và không lo false positive. |
| **5. Bạo lực (Violence / Fight)** | Action Recognition / Motion + Interaction | **Cách 1**: Fine-tune YOLO detect class `fight`/`violence` (dựa trên bounding box 2 người giằng co/đấm đá).<br>**Cách 2**: Dùng mô hình Action/Pose (VideoMAE hoặc ST-GCN/SlowFast). | Khuyến nghị giai đoạn 1 (trước 30/10): Dùng YOLOv8/v11 train trên dataset bạo lực + bounding box overlap để chạy realtime. |

---

## 2. KIẾN TRÚC TỔNG THỂ HỆ THỐNG (PIPELINE)

```
[RTSP Camera Stream / Video Feed]
               │
               ▼
      [Frame Extraction]
               │
       ┌───────┴────────────────────────┐
       ▼                                ▼
[Module 1: YOLOv8/v11 Multi-class]   [Module 2: ByteTrack & ROI Tracker]
 • Person                             • Theo dõi ID người
 • Phone / Calling                    • Kiểm tra người trong Chốt trực (ROI)
 • Cigarette / Smoking                • Đếm thời gian vắng mặt (Bỏ vị trí)
 • Violence / Fight                   
       │                                │
       └───────┬────────────────────────┘
               ▼
[Module 3: Temporal & Logic Filter (Tránh báo động giả)]
 • Ngủ gục: Tư thế gục + Đứng yên > 30s
 • Dùng điện thoại: Bbox phone liên tục > 5s
 • Hút thuốc: Bbox thuốc liên tục > 3s
 • Bỏ vị trí: Vùng trực trống > 2 phút
 • Bạo lực: Action 'Fight' xác suất > 75% trong 3 frames liên tiếp
               │
               ▼
[Module 4: Alert & Dashboard System]
 • Ghi log sự kiện (Timestamp, Camera ID, Ảnh chụp snapshot)
 • Cảnh báo âm thanh / Telegram / Web Dashboard
```

---

## 3. NGUỒN DATASET CHUẨN ĐỂ RETRAIN TRÊN KAGGLE

Bạn không cần tự gán nhãn từ đầu cho toàn bộ 5 hành vi, mà hãy tận dụng các Dataset mở chất lượng cao trên **Kaggle** và **Roboflow Universe**:

1. **Dataset Hút thuốc & Điện thoại (Smoking & Phone Use)**:
   - Roboflow Universe: `Cigarette and Phone Detection`, `Driver Drowsiness and Distraction`, `Smoking Detection YOLOv8`.
   - Kaggle: *Smoking and Calling Dataset*, *Driver Distraction Dataset*.
2. **Dataset Ngủ gật (Sleeping / Drowsy)**:
   - Roboflow Universe: `Sleeping Detection`, `Person Sleeping Posture`.
   - Kaggle: *Drowsiness Detection Dataset*.
3. **Dataset Bạo lực (Violence / Fighting)**:
   - Kaggle: *RWF-2000 (Real-World Fighting)*, *UCF-Crime Fight*, *Surveillance Violence Detection*.
   - Roboflow: *Fight Detection YOLOv8*.

---

## 4. LỘ TRÌNH 4 TUẦN ĐẾN HẠN 30/10

| Tuần | Thời gian | Nội dung công việc | Đầu ra (Deliverable) |
| :--- | :--- | :--- | :--- |
| **Tuần 1** | 01/10 – 07/10 | • Chuẩn bị tài khoản Kaggle, cấu hình API Roboflow.<br>• Tải & gộp 3 bộ dataset cốt lõi: Điện thoại, Hút thuốc, Bạo lực.<br>• Thống nhất danh sách class: `person`, `phone`, `smoking`, `sleeping`, `violence`. | Bộ dataset hoàn chỉnh định dạng YOLO (train/val/test) sẵn sàng trên Kaggle. |
| **Tuần 2** | 08/10 – 15/10 | • Chạy train/retrain mô hình YOLOv8m hoặc YOLOv11m trên Kaggle GPU.<br>• Đánh giá mAP@50, Precision, Recall.<br>• Tối ưu hyperparameters (epochs=50-100, imgsz=640). | File trọng số `best.pt` và biểu đồ đánh giá chất lượng. |
| **Tuần 3** | 16/10 – 23/10 | • Xây dựng logic **Bỏ vị trí trực** bằng ByteTrack + Vùng đa giác (ROI).<br>• Xây dựng bộ đếm thời gian (Temporal Window) lọc cảnh báo giả cho Ngủ và Dùng điện thoại.<br>• Test pipeline trên video quay thử nghiệm tình huống bảo vệ thực tế. | Script Python `inference_pipeline.py` chạy thử nghiệm trên video camera thực tế. |
| **Tuần 4** | 24/10 – 30/10 | • Tối ưu tốc độ: Export model sang ONNX / TensorRT / OpenVINO (đạt $\ge 20$ FPS).<br>• Đóng gói script nhận luồng RTSP camera mục tiêu.<br>• Viết báo cáo nghiệm thu, làm video demo 5 hành vi trình diễn cho sếp. | Báo cáo hoàn chỉnh + Video Demo + Mã nguồn triển khai. |
