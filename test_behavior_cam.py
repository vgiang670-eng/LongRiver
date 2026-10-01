"""
SCRIPT KIỂM TRA ĐỘ NHẠY MODEL BEHAVIOR (best.pt) TRỰC TIẾP
Dùng để test: Hút thuốc, Dùng điện thoại, Ngủ gật, Ăn uống
Hiển thị toàn bộ những gì AI nhìn thấy kèm độ tin cậy Conf
"""

import cv2
import time
from ultralytics import YOLO

def test_camera():
    model_path = 'best.pt'
    print(f"[*] Đang tải model: {model_path}")
    model = YOLO(model_path)
    print(f"[*] Các class của model: {model.names}")

    # Mở webcam với DirectShow để không bị delay khung hình
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    # Đặt độ phân giải nhẹ để CPU chạy siêu mượt 15-20 FPS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("\n" + "="*50)
    print("🚀 BẮT ĐẦU TEST WEBCAM - NHẤN 'q' ĐỂ THOÁT")
    print("Thử: Cầm điện thoại, ngậm điếu thuốc, đưa tay lên miệng, cúi gục đầu")
    print("="*50 + "\n")

    # Bảng màu cho từng hành vi
    colors = {
        'smoking': (0, 0, 255),    # Đỏ
        'phone': (0, 165, 255),    # Cam
        'sleeping': (255, 0, 255),  # Tím
        'eating': (0, 255, 255)    # Vàng
    }

    prev_time = time.time()

    while cap.isOpened():
        # Xóa sạch frame cũ trong hàng đợi để lấy đúng frame mới nhất (chống delay)
        cap.grab()
        ret, frame = cap.retrieve()
        if not ret:
            break

        # Chạy suy luận trên ảnh sạch nguyên bản với conf siêu nhạy (0.15)
        results = model(frame, conf=0.15, imgsz=480, verbose=False)[0]

        # Tính FPS
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

            # Chọn màu
            color = colors.get(cls_name, (0, 255, 0))

            # Vẽ khung dày nổi bật
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            label = f"{cls_name.upper()}: {conf:.2f}"
            
            # Vẽ nền chữ cho dễ đọc
            (w_txt, h_txt), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x1, y1 - 25), (x1 + w_txt, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # In log ra terminal nếu có phát hiện
        if detected_list:
            print(f"[AI PHÁT HIỆN]: {' | '.join(detected_list)}")

        # Hiển thị FPS lên màn hình
        cv2.putText(frame, f"FPS: {fps:.1f} | Conf Ngưỡng: 0.15", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("TEST NHAN DIEN HANH VI (BEST.PT)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_camera()
