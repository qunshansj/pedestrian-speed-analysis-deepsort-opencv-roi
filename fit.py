python


class PedestrianSpeedEstimator:
    def __init__(self, yolo_weights, deepsort_weights, video_path):
        self.yolo = YOLODetector(yolo_weights)
        self.deepsort = DeepSort(deepsort_weights)
        self.video_path = video_path

    def detect_and_track(self):
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        if not ret:
            print("Failed to read video")
            return

        roi = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
        cv2.destroyAllWindows()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            roi_frame = frame[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]

            detections = self.yolo.detect(roi_frame)

            tracker_outputs = self.deepsort.update(detections)

            for track in tracker_outputs:
                if track.is_confirmed() and track.time_since_update > 1:
                    continue

                speed = self.calculate_speed(track)

                bbox = track.to_tlbr()
                cv2.rectangle(roi_frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
                cv2.putText(roi_frame, f"ID: {track.track_id}, Speed: {speed:.2f} m/s",
                            (int(bbox[0]), int(bbox[1]) - 10), 0, 0.5, (255, 0, 0), 2)

            frame[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])] = roi_frame
            cv2.rectangle(frame, (int(roi[0]), int(roi[1])), (int(roi[0] + roi[2]), int(roi[1] + roi[3])), (0, 255, 0),
                          2)

            cv2.imshow('Frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def calculate_speed(self, track):
        if len(track.history) < 2:
            return 0.0

        prev_bbox, curr_bbox = track.history[-2], track.history[-1]
        dy = (curr_bbox[1] + curr_bbox[3]) / 2 - (prev_bbox[1] + prev_bbox[3]) / 2
        dx = (curr_bbox[0] + curr_bbox[2]) / 2 - (prev_bbox[0] + prev_bbox[2]) / 2

        speed = (dy ** 2 + dx ** 2) ** 0.5

        return speed
