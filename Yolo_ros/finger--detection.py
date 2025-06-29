import torch
import numpy as np
import cv2
from time import time

class FingerDetection:
    

    def __init__(self, capture_index, model_path):
        
        self.capture_index = capture_index
        self.model = self.load_model(capture_index, model_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)
        self.prev_center = None  # Previous center coordinates
        self.path = []  # List to store centroid path

    def get_video_capture(self):
       
        return cv2.VideoCapture(self.capture_index)

    def load_model(self, capture_index, model_path):
        
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
        return model

    def score_frame(self, frame):
        
        self.model.to(self.device)
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def plot_boxes(self, results, frame):
        
        labels, cord = results
        n = len(labels)
        cord2 = cord * 415  # Scale the coordinates if necessary

        # Initialize center color
        center_color = (0, 0, 255)  # Red color for center point

        for i in range(n):
            row = cord2[i]
            x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])

            # Calculate center coordinates
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            print(f"Centroid X: {center_x}, Centroid Y: {center_y}")

            # Draw bounding box (green)
            bgr = (0, 255, 0)  # Green color for box
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)

            # Draw center point (red)
            cv2.circle(frame, (center_x, center_y), 5, center_color, -1)  # Draw a red filled circle at center
                        # Draw center point (red)
            cv2.circle(frame, (center_x, center_y), 5, center_color, -1)  # Draw a red filled circle at center
            if 0 <= center_x < frame.shape[1] and 0 <= center_y < frame.shape[0]:
                fingertip_detected = False
                # Update path
                self.path.append((center_x, center_y))
                self.consecutive_frames_without_detection = 0
            else:
                # Increment counter if fingertip not detected
                self.consecutive_frames_without_detection += 1

        # Draw path
        for p in range(1, len(self.path)):
            cv2.line(frame, self.path[p - 1], self.path[p], center_color, 2)

        return frame

    def __call__(self):
        """
        This function is called when the class is executed, it runs the loop to read the video frame by frame.
        :return: void
        """
        cap = self.get_video_capture()
        assert cap.isOpened()

        while True:
            ret, frame = cap.read()
            assert ret
            
            # Resize the frame to match the input size expected by the YOLOv5 model
            frame = cv2.resize(frame, (416, 416))
            
            start_time = time()
            results = self.score_frame(frame)
            frame = self.plot_boxes(results, frame)
            end_time = time()

            fps = 1 / np.round(end_time - start_time, 2)
            cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            cv2.imshow('Finger Tip Detection', frame)


            if cv2.waitKey(5) & 0xFF == 27:
                break

        cap.release()

# Create a new object and execute.
detector = FingerDetection(capture_index=0, model_path='/home/prachi/Downloads/besttt.pt')
detector()