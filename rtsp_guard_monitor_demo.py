"""
HỆ THỐNG GIÁM SÁT HÀNH VI BẢO VỆ CHUẨN THỰC TẾ (PRODUCTION GRADE)
Bao gồm:
1. Hút thuốc (Smoking)
2. Sử dụng điện thoại (Phone usage)
3. Ngủ gật (Sleeping)
4. Ăn uống trong giờ trực (Eating)
5. Bỏ vị trí trực (Leaving guard post - Zone ROI & Timer)
"""

import cv2
import numpy as np
import time
import os
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
        
        # Lưu thời điểm cuối cùng nhìn thấy từng hành vi
        self.last_seen_time = {
            'smoking': 0.0,
            'phone': 0.0,
            'sleeping': 0.0,
            'eating': 0.0
        }
        # Thời gian giữ cảnh báo sau khi hết hành vi (2.0s rồi tắt ngay, không bị dính)
        self.alert_duration = 2.0

        self.frame_count = 0
        self.guard_in_roi = True
        self.last_person_boxes = []

        # Cấu hình nhãn và màu sắc
        self.behavior_config = {
            0: {'key': 'smoking', 'name': 'HUT THUOC', 'color': (0, 0, 255), 'alert': 'CANH BAO: DANG HUT THUOC LA!'},
            1: {'key': 'eating', 'name': 'AN UONG', 'color': (0, 255, 255), 'alert': 'CANH BAO: AN UONG TRONG GIO TRUC!'},
            2: {'key': 'sleeping', 'name': 'NGU GAT', 'color': (255, 0, 255), 'alert': 'CANH BAO: NGU GAT TRONG GIO TRUC!'},
            3: {'key': 'phone', 'name': 'DIEN THOAI', 'color': (0, 165, 255), 'alert': 'CANH BAO: SU DUNG DIEN THOAI!'}
        }

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

        # 2. SUY LUẬN HÀNH VI VỚI ĐỘ PHÂN GIẢI CHUẨN 640
        behavior_results = self.behavior_model(clean_input, conf=0.15, imgsz=640, verbose=False)[0]

        detected_in_this_frame = set()

        for box in behavior_results.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            if cls_id in self.behavior_config:
                cfg = self.behavior_config[cls_id]
                b_key = cfg['key']
                b_name = cfg['name']
                color = cfg['color']

                # Cập nhật thời điểm nhìn thấy hành vi này
                self.last_seen_time[b_key] = current_time
                detected_in_this_frame.add(b_key)

                # Vẽ khung viền dày dặn và nhãn nổi bật
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                label_txt = f"{b_name}: {conf:.2f}"
                (wt, ht), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                cv2.rectangle(frame, (x1, y1 - 24), (x1 + wt, y1), color, -1)
                cv2.putText(frame, label_txt, (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0) if b_key == 'eating' else (255, 255, 255), 2)
                print(f"🚨 [AI PHÁT HIỆN]: {label_txt}")

        # 3. QUÉT NGƯỜI TRỰC BẢO VỆ (1 lần mỗi 4 frame để giảm tải CPU)
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

        # Nếu phát hiện bất kỳ hành vi nào thì chắc chắn bảo vệ có mặt
        if detected_in_this_frame:
            self.guard_in_roi = True

        # 4. XỬ LÝ CẢNH BÁO
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

        # 5. CẢNH BÁO HÀNH VI: TỰ ĐỘNG TẮT SAU 2.0 GIÂY NẾU KHÔNG CÒN HÀNH VI
        for cls_id, cfg in self.behavior_config.items():
            b_key = cfg['key']
            elapsed_since_seen = current_time - self.last_seen_time[b_key]
            # Nếu hành vi vừa diễn ra trong vòng 2.0s gần nhất -> Hiện cảnh báo
            # Quá 2.0s không thấy -> Tự động biến mất ngay lập tức!
            if elapsed_since_seen < self.alert_duration:
                alerts.append(cfg['alert'])

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
    print("-> Nhan dien: 1. Hut thuoc  2. Dien thoai  3. Ngu gat  4. An uong")
    print("-> Chống dính cảnh báo: Tự động tắt sau 2 giây khi hết hành vi")
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
