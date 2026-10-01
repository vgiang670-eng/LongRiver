"""
SCRIPT KIỂM TRA ĐỘ NHẠY MODEL BEHAVIOR TRỰC TIẾP TRÊN GPU INTEL / CUDA
Dùng để test: Hút thuốc, Dùng điện thoại, Ngủ gật, Ăn uống
Hiển thị toàn bộ những gì AI nhìn thấy kèm độ tin cậy Conf và FPS
"""

import os
import cv2
import time
from ultralytics import YOLO

def detect_device_and_model():
    """Tự động phát hiện GPU (Intel Iris Xe hoặc NVIDIA) để tối ưu hóa, không ăn CPU"""
    try:
        import openvino as ov
        core = ov.Core()
        if 'GPU' in core.available_devices:
            if not os.path.exists('best_openvino_model') and os.path.exists('best.pt'):
                print("⚡ [TỐI ƯU HÓA]: Tự động chuyển đổi mô hình sang OpenVINO cho Intel GPU...")
                YOLO('best.pt').export(format='openvino')
            if os.path.exists('best_openvino_model'):
                print("🚀 [PHẦN CỨNG]: Phát hiện card đồ họa Intel Iris Xe GPU! Kích hoạt chạy trên GPU.")
                return 'best_openvino_model', 'intel:gpu'
    except Exception:
        pass

    import torch
    if torch.cuda.is_available():
        print("🚀 [PHẦN CỨNG]: Phát hiện NVIDIA CUDA GPU!")
        return 'best.pt', 0

    print("⚠️ [PHẦN CỨNG]: Chạy trên CPU.")
    return 'best.pt', 'cpu'

def test_camera():
    model_path, device = detect_device_and_model()
    print(f"[*] Đang nạp model: {model_path} trên thiết bị: {device}")
    model = YOLO(model_path)
    print(f"[*] Các class của model: {model.names}")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("\n" + "="*50)
    print(f"🚀 BẮT ĐẦU TEST WEBCAM TRÊN [{str(device).upper()}] - NHẤN 'q' ĐỂ THOÁT")
    print("="*50 + "\n")

    colors = {
        'smoking': (0, 0, 255),    # Đỏ
        'phone': (0, 165, 255),    # Cam
        'sleeping': (255, 0, 255),  # Tím
        'eating': (0, 255, 255)    # Vàng
    }

    prev_time = time.time()

    while cap.isOpened():
        cap.grab()
        ret, frame = cap.retrieve()
        if not ret:
            break

        # Chạy suy luận trực tiếp trên GPU
        predict_kwargs = {'conf': 0.15, 'imgsz': 640 if 'openvino' in model_path else 480, 'verbose': False}
        if device != 'cpu':
            predict_kwargs['device'] = device

        results = model.predict(frame, **predict_kwargs)[0]

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        detected_list = []
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            cls_name = model.names[cls_id].lower()
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            detected_list.append(f"{cls_name} ({conf:.2f})")
            color = colors.get(cls_name, (0, 255, 0))

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            label = f"{cls_name.upper()}: {conf:.2f}"
            (w_txt, h_txt), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x1, y1 - 25), (x1 + w_txt, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if detected_list:
            print(f"[AI PHÁT HIỆN]: {' | '.join(detected_list)}")

        cv2.putText(frame, f"Device: {str(device).upper()} | FPS: {fps:.1f}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("TEST NHAN DIEN HANH VI (GPU ACCELERATED)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_camera()
