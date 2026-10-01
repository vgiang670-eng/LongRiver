# HƯỚNG DẪN TỔNG HỢP DATASET ĐỂ HUẤN LUYỆN TRÊN KAGGLE

Để hoàn thành trước deadline **30/10**, bạn không nên gán nhãn thủ công từ số 0 mà hãy tận dụng các bộ dữ liệu cộng đồng đã được chuẩn hóa theo định dạng **YOLO** (gồm ảnh `.jpg` và file nhãn `.txt`).

---

## 1. CÁC NGUỒN DATASET MIỄN PHÍ CÓ SẴN TRÊN KAGGLE (CHỈ CẦN BẤM "ADD INPUT")

### 1.1. Dataset Hút thuốc, Dùng điện thoại & Ngủ gật (Đã có sẵn dạng YOLO)
* **`YOLO Dataset Smoking, Eating, Sleeping, Phone`**: Dataset tổng hợp sẵn cả 4 class: hút thuốc, ăn uống, ngủ gật và dùng điện thoại.
* **`Smoking and Drinking Dataset for YOLO`**: Dataset chuyên phát hiện điếu thuốc, hành vi hút thuốc.
* **`Driver Drowsiness and Distraction Dataset`**: Chứa hàng nghìn ảnh thực tế về người ngồi trực cúi gục đầu ngủ, áp điện thoại vào tai, cầm điện thoại nhắn tin.
* **`Yolo-smoke-detection`**: Hơn 2.000 ảnh đã gán nhãn YOLOv8 về hành vi hút thuốc.

### 1.2. Dataset Bạo lực / Xô xát (Violence / Fight Detection)
* **`Real Life Violence Situations Dataset`** (Tác giả: *mohamedmustafa*): Dataset nổi tiếng nhất trên Kaggle với 2.000 video camera giám sát thực tế (1.000 bạo lực + 1.000 bình thường).
* **`CCTV Aggressive Poses & Fight Detection`**: Dataset trích xuất từ camera góc cao chuyên phát hiện đánh nhau, xô xát dạng Bounding Box & Pose.
* **`Violence vs. Non-Violence Images`**: Hơn 11.000 frame ảnh đã phân loại sẵn để train nhận diện đánh nhau.

### 1.3. Hành vi Bỏ vị trí trực (Leaving Post)
* **TUYỆT ĐỐI KHÔNG CẦN TẢI DATASET RIÊNG**: 
  - Class `person` (người) đã được train sẵn trên tập dữ liệu chuẩn COCO (80 class) của YOLO.
  - Sử dụng logic vẽ vùng chốt trực (ROI Polygon) trong script `rtsp_guard_monitor_demo.py` đã cung cấp. Khi không có `person` trong vùng quá 1-2 phút thì tự động cảnh báo.

---

## 2. CÁCH GỘP (MERGE) DATASET TRÊN KAGGLE NHANH NHẤT

Để mô hình nhận diện được cả 4 hành vi phát hiện (`phone`, `smoking`, `sleeping`, `violence`) và `person`:

### Bước 1: Chuẩn hóa nhãn (Class Mapping)
Đảm bảo các file `.txt` nhãn có ID đồng nhất:
- `0`: `person`
- `1`: `phone`
- `2`: `smoking`
- `3`: `sleeping`
- `4`: `violence`

### Bước 2: Tạo cấu trúc thư mục trên Kaggle
```text
/kaggle/working/dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

### Bước 3: Cấu hình `data.yaml`
```yaml
path: /kaggle/working/dataset
train: images/train
val: images/val

names:
  0: person
  1: phone
  2: smoking
  3: sleeping
  4: violence
```
