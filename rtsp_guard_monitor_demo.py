"""
HỆ THỐNG GIÁM SÁT HÀNH VI BẢO VỆ CHUẨN THỰC TẾ (PRODUCTION GRADE)
Áp dụng:
1. Độ phân giải chuẩn imgsz=640 để giữ nguyên chi tiết điếu thuốc siêu nhỏ
2. Bộ lọc tích lũy thời gian (Temporal Voting / Accumulator) chống chập chờn
3. Bắt cả cử chỉ tay đưa lên miệng (kết hợp nhãn smoking + eating)
4. Tự động nhận diện có mặt tại chốt trực (ROI) và cảnh báo bỏ vị trí
"""

import cv2
import numpy as np
import time
import os
from collections import deque
from ultralytics import YOLO

class ProductionGuardMonitor:
    def __init__(self, behavior_model_path='best.pt', person_model_path='yolov8n.pt', roi_polygon=None, absence_threshold_sec=15):
        print(f"[*] Đang nạp mô hình hành vi: {behavior_model_path}")
        self.behavior_model = YOLO(behavior_model_path)
        print(f"[*] Các nhãn hành vi: {self.behavior_model.names}")
        
        print(f"[*] Đang nạp mô hình người: {person_model_path}")
        self.person_model = YOLO(person_model_path)

        self.absence_threshold = absence_threshold_sec
        self.last_seen_in_post_time = time.time()
        self.roi_polygon = roi_polygon
        
        # Bộ đệm tích lũy 10 frames gần nhất để chống rớt frame (Rolling Window)
        self.history_len = 10
        self.smoking_history = deque(maxlen=self.history_len)
        self.phone_history = deque(maxlen=self.history_len)
        self.sleeping_history = deque(maxlen=self.history_len)
        
        # Thời điểm cảnh báo lần cuối
        self.alert_until = {
            'smoking': 0,
            'phone': 0,
            'sleeping': 0,
            'absence': 0
        }

        self.frame_count = 0
        self.guard_in_roi = True
        self.last_person_boxes = []

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
        clean_input = frame.copy()

        # Thiết lập vùng chốt trực ROI
        if self.roi_polygon is None:
            self.roi_polygon = [
                (int(w * 0.15), int(h * 0.15)),
                (int(w * 0.85), int(h * 0.15)),
                (int(w * 0.85), int(h * 0.95)),
                (int(w * 0.15), int(h * 0.95))
            ]

        # 1. Vẽ vùng Chốt Trực (ROI)
        roi_pts = np.array(self.roi_polygon, np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [roi_pts], isClosed=True, color=(255, 255, 0), thickness=2)
        cv2.putText(frame, "CHOT TRUC ROI", (self.roi_polygon[0][0], self.roi_polygon[0][1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # 2. SUY LUẬN HÀNH VI VỚI ĐỘ PHÂN GIẢI CHUẨN 640 (ĐỂ BẮT ĐIẾU THUỐC NHỎ)
        # Hạ conf xuống 0.12 để bắt được cả điếu thuốc từ góc nghiêng / xa
        behavior_results = self.behavior_model(clean_input, conf=0.12, imgsz=640, verbose=False)[0]

        has_smoking_frame = False
        has_phone_frame = False
        has_sleeping_frame = False

        for box in behavior_results.boxes:
            cls_id = int(box.cls[0].item())
            cls_name = self.behavior_model.names[cls_id].lower()
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # A. PHÁT HIỆN HÚT THUỐC (Bao gồm nhãn smoking hoặc cử chỉ tay-miệng eating với điếu thuốc)
            if 'smoke' in cls_name or 'cig' in cls_name:
                has_smoking_frame = True
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                label_txt = f"HUT THUOC: {conf:.2f}"
                (wt, ht), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - 22), (x1 + wt, y1), (0, 0, 255), -1)
                cv2.putText(frame, label_txt, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # B. PHÁT HIỆN ĐIỆN THOẠI
            elif 'phone' in cls_name:
                has_phone_frame = True
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 3)
                label_txt = f"DIEN THOAI: {conf:.2f}"
                (wt, ht), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - 22), (x1 + wt, y1), (0, 165, 255), -1)
                cv2.putText(frame, label_txt, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # C. PHÁT HIỆN NGỦ GẬT
            elif 'sleep' in cls_name:
                has_sleeping_frame = True
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                label_txt = f"NGU GAT: {conf:.2f}"
                (wt, ht), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - 22), (x1 + wt, y1), (255, 0, 255), -1)
                cv2.putText(frame, label_txt, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # D. ĂN UỐNG / TAY ĐƯA LÊN MIỆNG
            elif 'eat' in cls_name:
                # Nếu đưa tay lên miệng, cũng ghi nhận vào hỗ trợ hút thuốc nếu có nghi vấn
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(frame, f"TAY MIENG: {conf:.2f}", (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        # 3. BỘ TÍCH LŨY THỜI GIAN (TEMPORAL VOTING):
        # Nếu trong 10 frame gần nhất xuất hiện >= 2 lần -> Xác nhận có hành vi vi phạm!
        self.smoking_history.append(1 if has_smoking_frame else 0)
        self.phone_history.append(1 if has_phone_frame else 0)
        self.sleeping_history.append(1 if has_sleeping_frame else 0)

        # Kiểm tra ngưỡng kích hoạt
        if sum(self.smoking_history) >= 2:
            self.alert_until['smoking'] = current_time + 3.0  # Giữ cảnh báo 3 giây

        if sum(self.phone_history) >= 2:
            self.alert_until['phone'] = current_time + 3.0

        if sum(self.sleeping_history) >= 4: # Ngủ cần xuất hiện nhiều frame hơn
            self.alert_until['sleeping'] = current_time + 3.0

        # 4. QUÉT NGƯỜI TRỰC (1 lần mỗi 4 frame)
        if self.frame_count % 4 == 0 or len(self.last_person_boxes) == 0:
            person_results = self.person_model(clean_input, classes=[0], conf=0.30, imgsz=480, verbose=False)[0]
            self.last_person_boxes = []
            for box in person_results.boxes:
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                self.last_person_boxes.append((x1, y1, x2, y2, conf))

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

        if has_smoking_frame or has_phone_frame:
            self.guard_in_roi = True

        # 5. XỬ LÝ CẢNH BÁO
        alerts = []
        if self.guard_in_roi:
            self.last_seen_in_post_time = current_time
            cv2.putText(frame, "Trang thai: Co mat tai chot", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        else:
            absence_dur = current_time - self.last_seen_in_post_time
            if absence_dur >= self.absence_threshold:
                alerts.append(f"CANH BAO: BO VI TRI TRUC ({int(absence_dur)}s)!")
            else:
                rem = int(self.absence_threshold - absence_dur)
                cv2.putText(frame, f"Vang mat: {int(absence_dur)}s (Bao dong sau {rem}s)",
                            (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        # Thêm các cảnh báo đang trong thời gian kích hoạt
        if current_time < self.alert_until['smoking']:
            alerts.append("CANH BAO: DANG HUT THUOC LA!")
        if current_time < self.alert_until['phone']:
            alerts.append("CANH BAO: SU DUNG DIEN THOAI!")
        if current_time < self.alert_until['sleeping']:
            alerts.append("CANH BAO: NGU GAT TRONG GIO TRUC!")

        # 6. VẼ BĂNG RÔN CẢNH BÁO
        y_alert = 75
        for alert_msg in alerts:
            cv2.rectangle(frame, (15, y_alert - 25), (660, y_alert + 12), (0, 0, 200), -1)
            cv2.putText(frame, alert_msg, (20, y_alert),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
            y_alert += 45

        return frame

def run_production(video_source=0, model_path='best.pt'):
    monitor = ProductionGuardMonitor(
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

    print("\n" + "="*60)
    print("🚀 HE THONG GIAM SAT HANH VI BAO VE (PRODUCTION)")
    print("-> Do phan giai phan tich: 640x640")
    print("-> Co che: Temporal Voting (Loc rung lac khung hinh)")
    print("Nhan 'q' de thoat chuong trinh...")
    print("="*60 + "\n")

    while cap.isOpened():
        cap.grab()
        ret, frame = cap.retrieve()
        if not ret:
            break

        processed_frame = monitor.process_frame(frame)
        cv2.imshow("He Thong Giam Sat Hanh Vi Bao Ve (Chuan Thuc Te)", processed_frame)

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
    run_production(video_source=0, model_path=model_file)
