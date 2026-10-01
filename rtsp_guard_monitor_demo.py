"""
HỆ THỐNG GIÁM SÁT HÀNH VI BẢO VỆ CHUẨN THỰC TẾ (PRODUCTION GRADE)
Bao gồm:
1. Hút thuốc (Smoking)
2. Sử dụng điện thoại (Phone usage)
3. Ngủ gật (Sleeping) - Phải duy trì liên tục > 5 giây mới báo, chống báo giả khi nhìn xuống
4. Ăn uống trong giờ trực (Eating)
5. Bỏ vị trí trực (Leaving guard post - Zone ROI & Timer)
"""

import cv2
import numpy as np
import time
import os
from ultralytics import YOLO

def detect_hardware():
    """Tự động phát hiện Intel Iris Xe GPU hoặc NVIDIA CUDA để giải phóng CPU"""
    try:
        import openvino as ov
        core = ov.Core()
        if 'GPU' in core.available_devices:
            if not os.path.exists('best_openvino_model') and os.path.exists('best.pt'):
                print("⚡ [TỐI ƯU HÓA]: Tự động chuyển đổi mô hình sang OpenVINO cho Intel GPU...")
                YOLO('best.pt').export(format='openvino')
            if not os.path.exists('yolov8n_openvino_model') and os.path.exists('yolov8n.pt'):
                YOLO('yolov8n.pt').export(format='openvino')
            if os.path.exists('best_openvino_model'):
                print("🚀 [PHẦN CỨNG]: Đã kích hoạt Intel Iris Xe GPU thông qua OpenVINO!")
                b_model = 'best_openvino_model'
                p_model = 'yolov8n_openvino_model' if os.path.exists('yolov8n_openvino_model') else 'yolov8n.pt'
                return b_model, p_model, 'intel:gpu'
    except Exception:
        pass
    import torch
    if torch.cuda.is_available():
        print("🚀 [PHẦN CỨNG]: Đã kích hoạt NVIDIA CUDA GPU!")
        return 'best.pt', 'yolov8n.pt', 0
    print("⚠️ [PHẦN CỨNG]: Chạy trên CPU.")
    return 'best.pt', 'yolov8n.pt', 'cpu'

class ProductionGuardMonitor:
    def __init__(self, behavior_model_path=None, person_model_path=None, roi_polygon=None, absence_threshold_sec=15):
        def_b, def_p, dev = detect_hardware()
        self.behavior_model_path = behavior_model_path or def_b
        self.person_model_path = person_model_path or def_p
        self.device = dev

        print(f"[*] Đang nạp mô hình hành vi: {self.behavior_model_path} [Device: {self.device}]")
        self.behavior_model = YOLO(self.behavior_model_path)
        print(f"[*] Các nhãn hành vi: {self.behavior_model.names}")
        
        print(f"[*] Đang nạp mô hình người: {self.person_model_path} [Device: {self.device}]")
        self.person_model = YOLO(self.person_model_path)

        self.absence_threshold = absence_threshold_sec
        self.last_seen_in_post_time = time.time()
        self.roi_polygon = roi_polygon
        
        # Lưu thời điểm cuối cùng nhìn thấy từng hành vi
        self.last_seen_time = {
            'smoking': 0.0,
            'phone': 0.0,
            'eating': 0.0
        }
        self.alert_duration = 1.0  # Tắt cảnh báo sau 1.0s khi hết hành vi

        # BỘ ĐẾM THỜI GIAN RIÊNG CHO NGỦ GẬT (Phải gục đầu liên tục > 5s mới báo động)
        self.sleep_start_time = None
        self.sleep_required_seconds = 5.0
        self.is_sleeping_alert = False

        self.frame_count = 0
        self.guard_in_roi = True
        self.last_person_boxes = []

        # Ngưỡng tin cậy (Confidence) chuẩn hóa để chống báo động giả
        self.conf_thresholds = {
            0: 0.35,  # Hút thuốc: >= 0.35
            1: 0.50,  # Ăn uống: nâng lên 0.50 (tránh nhầm khi đưa tay/cầm điện thoại gần mặt)
            2: 0.40,  # Ngủ gật: >= 0.40
            3: 0.35   # Điện thoại: >= 0.35
        }

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

        # 2. SUY LUẬN HÀNH VI
        predict_kwargs = {'conf': 0.15, 'imgsz': 640 if 'openvino' in str(self.behavior_model_path) else 480, 'verbose': False}
        if self.device != 'cpu':
            predict_kwargs['device'] = self.device
        behavior_results = self.behavior_model.predict(clean_input, **predict_kwargs)[0]

        detected_in_this_frame = set()

        # Bước 1: Thu thập tất cả các bounding box hợp lệ đạt ngưỡng tin cậy
        valid_candidates = []
        for box in behavior_results.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            required_conf = self.conf_thresholds.get(cls_id, 0.35)
            if conf < required_conf:
                continue

            # Lọc bỏ bounding box rác quá lớn
            box_w = x2 - x1
            box_h = y2 - y1
            if cls_id == 0 and (box_w > w * 0.45 or box_h > h * 0.45):
                continue

            valid_candidates.append({
                'cls_id': cls_id,
                'conf': conf,
                'box': (x1, y1, x2, y2)
            })

        # Bước 2: Khử xung đột cử chỉ tay:
        # Nếu đã phát hiện Điện thoại (cls 3) hoặc Hút thuốc (cls 0) -> LOẠI BỎ hoàn toàn nhãn Ăn uống (cls 1)
        has_phone_or_smoke = any(c['cls_id'] in (0, 3) for c in valid_candidates)
        if has_phone_or_smoke:
            valid_candidates = [c for c in valid_candidates if c['cls_id'] != 1]
            self.last_seen_time['eating'] = 0.0  # Tắt ngay lập tức mọi cảnh báo ăn uống còn vương lại

        # Bước 3: Vẽ và cập nhật thời điểm nhìn thấy
        for c in valid_candidates:
            cls_id = c['cls_id']
            conf = c['conf']
            x1, y1, x2, y2 = c['box']

            if cls_id in self.behavior_config:
                cfg = self.behavior_config[cls_id]
                b_key = cfg['key']
                b_name = cfg['name']
                color = cfg['color']

                detected_in_this_frame.add(b_key)
                if b_key != 'sleeping':
                    self.last_seen_time[b_key] = current_time

                # Vẽ khung viền và nhãn nổi bật
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                label_txt = f"{b_name}: {conf:.2f}"
                (wt, ht), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                cv2.rectangle(frame, (x1, y1 - 24), (x1 + wt, y1), color, -1)
                cv2.putText(frame, label_txt, (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0) if b_key == 'eating' else (255, 255, 255), 2)

        # 3. LOGIC RIÊNG CHO NGỦ GẬT (Phải gục đầu liên tục > 5s mới báo động)
        alerts = []
        if 'sleeping' in detected_in_this_frame:
            if self.sleep_start_time is None:
                self.sleep_start_time = current_time
            sleep_elapsed = current_time - self.sleep_start_time
            
            if sleep_elapsed >= self.sleep_required_seconds:
                self.is_sleeping_alert = True
                alerts.append(f"CANH BAO: NGU GAT TRONG GIO TRUC ({int(sleep_elapsed)}s)!")
            else:
                remaining_sleep = int(self.sleep_required_seconds - sleep_elapsed)
                cv2.putText(frame, f"Nghi van ngu gat: {int(sleep_elapsed)}s/{int(self.sleep_required_seconds)}s",
                            (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        else:
            # Ngẩng đầu dậy -> Reset ngay lập tức
            self.sleep_start_time = None
            self.is_sleeping_alert = False

        # 4. QUÉT NGƯỜI TRỰC BẢO VỆ (1 lần mỗi 4 frame)
        if self.frame_count % 4 == 0 or len(self.last_person_boxes) == 0:
            p_kwargs = {'classes': [0], 'conf': 0.30, 'imgsz': 480, 'verbose': False}
            if self.device != 'cpu':
                p_kwargs['device'] = self.device
            person_results = self.person_model.predict(clean_input, **p_kwargs)[0]
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

        # Nếu phát hiện bất kỳ hành vi nào trong frame thì chắc chắn bảo vệ có mặt
        if detected_in_this_frame:
            self.guard_in_roi = True

        # 5. LOGIC BỎ VỊ TRÍ TRỰC
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

        # 6. CẢNH BÁO CÁC HÀNH VI KHÁC: TỰ ĐỘNG TẮT SAU 2.0 GIÂY
        for cls_id in [0, 1, 3]:  # Hút thuốc, Ăn uống, Điện thoại
            cfg = self.behavior_config[cls_id]
            b_key = cfg['key']
            elapsed = current_time - self.last_seen_time[b_key]
            if elapsed < self.alert_duration:
                alerts.append(cfg['alert'])

        # 7. VẼ BĂNG RÔN CẢNH BÁO
        y_alert = 95
        for alert_msg in alerts:
            cv2.rectangle(frame, (15, y_alert - 25), (660, y_alert + 12), (0, 0, 200), -1)
            cv2.putText(frame, alert_msg, (20, y_alert),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
            y_alert += 45

        return frame

def run_production(video_source=0, model_path=None):
    monitor = ProductionGuardMonitor(
        behavior_model_path=model_path,
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
    print(f"-> Chay tren thiet bi: {str(monitor.device).upper()}")
    print("-> 1. Hut thuoc (Conf >= 0.35 + Loc box to)")
    print("-> 2. Dien thoai (Conf >= 0.35)")
    print("-> 3. An uong (Conf >= 0.30)")
    print("-> 4. Ngu gat (Conf >= 0.40 + Guc dau lien tuc > 5 giay moi bao)")
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
    run_production(video_source=0)
