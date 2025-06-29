#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
from time import time

class FingerDetection:


    def __init__(self, capture_index, model_path):
     
        self.capture_index = capture_index
        self.model = self.load_model(model_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.prev_center = None  # Previous center coordinates
        self.path = []  # List to store centroid path
        self.first_detection = True
        print("Using Device: ", self.device)
        self.world_width = 11  # Initialize here with the correct value
        self.world_height = 11
        # ROS initialization
        rospy.init_node('finger_detection', anonymous=True)
        self.bridge = CvBridge()

        # Create a publisher for the turtle control commands
        self.norm_centre_publisher = rospy.Publisher('/fingertip_coordinates', Point, queue_size=10)

        
    def get_video_capture(self):
   
        return cv2.VideoCapture(self.capture_index)

    def load_model(self, model_path):
  
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
        cord2 = cord * 415  
        center_color = (0, 0, 255)  
              
        for i in range(n):
            row = cord2[i]
            x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])

            # Calculate center coordinates
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Normalize the coordinates
            grid_size = 12 
            norm_y = (center_x / frame.shape[1]) * (grid_size - 1)
            norm_x = (center_y / frame.shape[0]) * (grid_size - 1)
            # Draw bounding box (green)
            bgr = (0, 255, 0)  # Green color for box
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)

            # Draw center point (red)
            cv2.circle(frame, (center_x, center_y), 5, center_color, -1) 

            if self.first_detection:
                self.path = [(center_x, center_y)] 
                self.first_detection = False 
            elif 0 <= center_x < frame.shape[1] and 0 <= center_y < frame.shape[0]:
                self.path.append((center_x, center_y)) 
            else:
                pass  
            msg = Point(x=norm_y, y=(11-norm_x))  
            print("Publishing:", msg) 
            self.norm_centre_publisher.publish(msg)

        # Draw path
        for p in range(1, len(self.path)):
            cv2.line(frame, self.path[p - 1], self.path[p], center_color, 2)
        return frame  

    def __call__(self):
        """
        Main execution function.
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