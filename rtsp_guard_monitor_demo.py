"""
HỆ THỐNG GIÁM SÁT HÀNH VI BẢO VỆ TẠI MỤC TIÊU (CAMERA AI)
Bao gồm 5 hành vi:
1. Ngủ gật (Sleeping)
2. Sử dụng điện thoại (Phone usage)
3. Hút thuốc (Smoking)
4. Bỏ vị trí trực (Leaving guard post - Zone ROI & Timer logic)
5. Bạo lực (Violence / Fight)
"""

import cv2
import numpy as np
import time
import os
from ultralytics import YOLO

class GuardBehaviorMonitor:
    def __init__(self, behavior_model_path='best.pt', person_model_path='yolov8n.pt', roi_polygon=None, absence_threshold_sec=15):
        print(f"[*] Đang tải mô hình hành vi từ: {behavior_model_path}")
        self.behavior_model = YOLO(behavior_model_path)
        print(f"[*] Các class của mô hình hành vi: {self.behavior_model.names}")
        
        print(f"[*] Đang tải mô hình phát hiện người từ: {person_model_path}")
        self.person_model = YOLO(person_model_path)

        self.absence_threshold = absence_threshold_sec
        self.last_seen_in_post_time = time.time()
        self.roi_polygon = roi_polygon
        
        # Bộ đếm khung hình để tối ưu hóa CPU
        self.frame_count = 0
        self.last_person_boxes = []
        self.guard_in_roi = True

        # Bảng màu sắc cho từng hành vi
        self.behavior_colors = {
            'smoking': (0, 0, 255),    # Đỏ
            'phone': (0, 165, 255),    # Cam
            'sleeping': (255, 0, 255),  # Tím
            'eating': (0, 255, 255),   # Vàng
            'violence': (0, 0, 255)
        }

        # Bộ đệm thời gian duy trì cảnh báo (Chống chớp tắt khi bị lọt 1-2 frame)
        self.last_detected_time = {
            'smoking': 0,
            'phone': 0,
            'sleeping': 0,
            'eating': 0,
            'violence': 0
        }
        self.alert_hold_seconds = 2.5  # Giữ cảnh báo hiển thị trong ít nhất 2.5 giây

    def is_inside_roi(self, point, polygon):
        if polygon is None or len(polygon) < 3:
            return True
        pt = (int(point[0]), int(point[1]))
        poly_np = np.array(polygon, dtype=np.int32)
        return cv2.pointPolygonTest(poly_np, pt, False) >= 0

    def process_frame(self, frame):
        current_time = time.time()
        h, w, _ = frame.shape
        self.frame_count += 1
        
        # Ảnh gốc sạch để suy luận AI
        clean_input = frame.copy()

        # Thiết lập vùng ROI chốt trực mặc định
        if self.roi_polygon is None:
            self.roi_polygon = [
                (int(w * 0.15), int(h * 0.2)),
                (int(w * 0.85), int(h * 0.2)),
                (int(w * 0.85), int(h * 0.95)),
                (int(w * 0.15), int(h * 0.95))
            ]

        # 1. Vẽ vùng Chốt Trực (ROI)
        roi_pts = np.array(self.roi_polygon, np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [roi_pts], isClosed=True, color=(255, 255, 0), thickness=2)
        cv2.putText(frame, "CHOT TRUC ROI", (self.roi_polygon[0][0], self.roi_polygon[0][1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # 2. PHÁT HIỆN HÀNH VI VI PHẠM (Chạy MỌI FRAME với logic siêu nhạy của test_behavior_cam)
        behavior_results = self.behavior_model(clean_input, conf=0.15, imgsz=480, verbose=False)[0]
        
        detected_this_frame = set()

        for box in behavior_results.boxes:
            cls_id = int(box.cls[0].item())
            cls_name = self.behavior_model.names[cls_id].lower()
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Cập nhật thời điểm phát hiện
            for b_key in ['smoking', 'phone', 'sleeping', 'eating']:
                if b_key in cls_name or (b_key == 'smoking' and 'cig' in cls_name):
                    self.last_detected_time[b_key] = current_time
                    detected_this_frame.add(b_key)
                    
                    color = self.behavior_colors.get(b_key, (0, 0, 255))
                    # Vẽ khung dày nổi bật giống test_behavior_cam
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    label_txt = f"{cls_name.upper()}: {conf:.2f}"
                    (w_t, h_t), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                    cv2.rectangle(frame, (x1, y1 - 25), (x1 + w_t, y1), color, -1)
                    cv2.putText(frame, label_txt, (x1, y1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                    print(f"🚨 [AI PHÁT HIỆN]: {label_txt}")

        # 3. PHÁT HIỆN NGƯỜI (Tối ưu CPU: Chỉ cần chạy 1 lần mỗi 4 frames)
        if self.frame_count % 4 == 0 or len(self.last_person_boxes) == 0:
            person_results = self.person_model(clean_input, classes=[0], conf=0.35, imgsz=480, verbose=False)[0]
            self.last_person_boxes = []
            for box in person_results.boxes:
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                self.last_person_boxes.append((x1, y1, x2, y2, conf))

        # Vẽ vị trí người
        self.guard_in_roi = False
        for (x1, y1, x2, y2, conf) in self.last_person_boxes:
            center_point = ((x1 + x2) / 2, (y1 + y2) / 2)
            if self.is_inside_roi(center_point, self.roi_polygon):
                self.guard_in_roi = True
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Bao ve dang truc ({conf:.2f})", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 180, 180), 1)

        # Nếu phát hiện bất kỳ hành vi nào trong frame thì chắc chắn bảo vệ có mặt
        if detected_this_frame:
            self.guard_in_roi = True

        # 4. LOGIC BỎ VỊ TRÍ TRỰC
        alerts = []
        if self.guard_in_roi:
            self.last_seen_in_post_time = current_time
            cv2.putText(frame, "Trang thai: Co mat tai chot",
                        (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        else:
            absence_duration = current_time - self.last_seen_in_post_time
            if absence_duration >= self.absence_threshold:
                alerts.append(f"CANH BAO: BO VI TRI TRUC ({int(absence_duration)}s)!")
            else:
                remaining = int(self.absence_threshold - absence_duration)
                cv2.putText(frame, f"Vang mat: {int(absence_duration)}s (Canh bao sau {remaining}s)",
                            (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        # 5. LOGIC HIỂN THỊ CẢNH BÁO HÀNH VI (Có bộ đệm chống chớp tắt 2.5s)
        labels_map = {
            'smoking': 'HUT THUOC LA',
            'phone': 'DUNG DIEN THOAI',
            'sleeping': 'NGU GAT',
            'eating': 'AN UONG TRONG GIO TRUC'
        }
        for b_key, label_name in labels_map.items():
            time_since_detect = current_time - self.last_detected_time[b_key]
            if time_since_detect < self.alert_hold_seconds:
                alerts.append(f"CANH BAO: {label_name}!")

        # 6. VẼ BĂNG CẢNH BÁO ĐỎ NỔI BẬT LÊN MÀN HÌNH
        y_alert = 75
        for alert_msg in alerts:
            cv2.rectangle(frame, (15, y_alert - 25), (660, y_alert + 12), (0, 0, 200), -1)
            cv2.putText(frame, alert_msg, (20, y_alert),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
            y_alert += 45

        return frame

def run_demo(video_source=0, model_path='best.pt'):
    monitor = GuardBehaviorMonitor(
        behavior_model_path=model_path,
        person_model_path='yolov8n.pt',
        absence_threshold_sec=15
    )
    cap = cv2.VideoCapture(video_source, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(video_source)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("[*] Khoi chay thanh cong! Nhan 'q' de thoat chuong trinh...")
    while cap.isOpened():
        cap.grab()
        ret, frame = cap.retrieve()
        if not ret:
            break

        processed_frame = monitor.process_frame(frame)
        cv2.imshow("He Thong Giam Sat Hanh Vi Bao Ve", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if os.path.exists('best.pt'):
        model_file = 'best.pt'
    elif os.path.exists('kaggle/working/runs/guard_behavior_model/weights/best.pt'):
        model_file = 'kaggle/working/runs/guard_behavior_model/weights/best.pt'
    else:
        model_file = 'yolov8n.pt'
        
    print(f"[*] Sử dụng file weights: {model_file}")
    run_demo(video_source=0, model_path=model_file)
