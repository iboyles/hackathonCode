import cv2
import numpy as np
import torch
import time
import logging
import insightface
import requests

# Suppress unnecessary logging
logging.getLogger().setLevel(logging.WARNING)

def detect_person(img, detector):
    """Detect people in the image using the provided detector."""
    bboxes, kpss = detector.detect(img)
    if bboxes is None or kpss is None:
        return np.array([]), np.array([])
    
    bboxes = np.round(bboxes[:, :4]).astype(int)
    kpss = np.round(kpss).astype(int)
    kpss[:, :, 0] = np.clip(kpss[:, :, 0], 0, img.shape[1])
    kpss[:, :, 1] = np.clip(kpss[:, :, 1], 0, img.shape[0])
    vbboxes = bboxes.copy()
    vbboxes[:, 0] = kpss[:, 0, 0]
    vbboxes[:, 1] = kpss[:, 0, 1]
    vbboxes[:, 2] = kpss[:, 4, 0]
    vbboxes[:, 3] = kpss[:, 4, 1]
    return bboxes, vbboxes

def detect_dangerous_objects(frame, model, dangerous_items):
    """Detect dangerous objects in the image using YOLOv5."""
    results = model(frame)
    dangerous_objects = []
    for detection in results.pred[0]:
        x1, y1, x2, y2, confidence, class_id = detection
        class_name = model.names[int(class_id)]
        if class_name in dangerous_items:
            dangerous_objects.append((int(x1), int(y1), int(x2), int(y2), class_name, float(confidence)))
    return dangerous_objects

def check_proximity_to_door(bbox, door_region):
    """Check if a person is near the door."""
    x1, y1, x2, y2 = bbox
    dx1, dy1, dx2, dy2 = door_region
    is_near = (x1 < dx2 and x2 > dx1 and y1 < dy2 and y2 > dy1)
    return is_near

def draw_door(frame, door_region):
    x1, y1, x2, y2 = door_region
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
    cv2.putText(frame, "Door", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

if __name__ == '__main__':
    # Load the face detection model
    detector = insightface.model_zoo.get_model('scrfd_person_2.5g.onnx', download=True)
    detector.prepare(0, nms_thresh=0.5, input_size=(640, 640))

    # Load the YOLOv5 model
    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)

    # Start video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    # Initialize entry/exit counters and related variables
    detection_threshold = 10
    no_detection_threshold = 20
    detection_count = 0
    no_detection_count = 0
    person_present = False
    entries = 0
    exits = 0

    # Capture first frame to set up door region
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Failed to capture first frame.")
        exit()

    # Allow user to draw door region
    door_region = cv2.selectROI("Select Door Region", first_frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Door Region")

    # Initialize dangerous object detection tracking
    dangerous_items = ['knife', 'firearm']  # Define dangerous items
    danger_detected = {item: 0 for item in dangerous_items}
    danger_threshold = 2  # Number of consecutive frames to detect

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Detect persons
        bboxes, vbboxes = detect_person(frame, detector)

        # Detect dangerous objects
        dangerous_objects = detect_dangerous_objects(frame, yolo_model, dangerous_items)

        # Draw door region
        draw_door(frame, door_region)

        # Check proximity to door and handle entry/exit logic
        for i, bbox in enumerate(bboxes):
            if not check_proximity_to_door(bbox, door_region):
                if not person_present:
                    detection_count += 1
                    no_detection_count = 0
                    if detection_count >= detection_threshold:
                        entries += 1
                        person_present = True
                        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())
                        print(f"Entry detected at {timestamp}")
                        requests.post('http://localhost:5000/detect', json={'action': 'enter', 'timestamp': timestamp})
                        break
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                cv2.putText(frame, "Away from Door", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            else:
                if person_present:
                    no_detection_count += 1
                    detection_count = 0
                    if no_detection_count >= no_detection_threshold:
                        exits += 1
                        person_present = False
                        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())
                        print(f"Exit detected at {timestamp}")
                        requests.post('http://localhost:5000/detect', json={'action': 'exit', 'timestamp': timestamp})
                        break
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
                cv2.putText(frame, "Near Door", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Draw detected dangerous objects and track detections
        for obj in dangerous_objects:
            x1, y1, x2, y2, class_name, confidence = obj
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            if class_name in dangerous_items:
                danger_detected[class_name] += 1
                if danger_detected[class_name] >= danger_threshold:
                    # Send API request for dangerous object detection
                    timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())
                    print(f"Dangerous object '{class_name}' detected at {timestamp}")
                    requests.post('http://localhost:5000/detect', json={'action': 'danger', 'object': class_name, 'timestamp': timestamp})
                    # Reset detection count for this object
                    danger_detected[class_name] = 0
            else:
                danger_detected[class_name] = 0  # Reset count if object is not dangerous

        # Display entry and exit counts on the frame
        cv2.putText(frame, f"Entries: {entries}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Exits: {exits}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the video feed
        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
