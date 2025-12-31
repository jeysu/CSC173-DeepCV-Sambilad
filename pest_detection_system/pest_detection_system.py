import cv2
from ultralytics import YOLO
import time

# Change video source for demo
VIDEO_SOURCE = 'videos\monkey.mp4' 
MODEL_PATH = 'model_and_weights.pt' 

PEST_CLASSES = ['Pig', 'Monkey'] 
LIVESTOCK_CLASSES = ['Cow', 'Chicken']

CONFIDENCE_THRESHOLD = 0.5
ALARM_COOLDOWN = 2.0

def main():
    print(f"Loading model from {MODEL_PATH}.")
    model = YOLO(MODEL_PATH)
    
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"Error: Could not open video source {VIDEO_SOURCE}.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    prev_time = 0
    last_alarm_time = 0
    
    print("Starting Inference. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream.")
            break
        
        frame = cv2.resize(frame, (1280, 720))
        height, width, _ = frame.shape 

        results = model(frame, stream=True, conf=CONFIDENCE_THRESHOLD)

        alarm_triggered = False
        detected_pests = []

        for r in results:
            boxes = r.boxes
            
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls_id]

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                color = (255, 0, 0)
                label_prefix = ""

                if class_name in PEST_CLASSES:
                    color = (0, 0, 255)
                    label_prefix = "PEST: "
                    alarm_triggered = True
                    detected_pests.append(class_name)
                    
                elif class_name in LIVESTOCK_CLASSES:
                    color = (0, 255, 0)
                    label_prefix = "SAFE: "

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{label_prefix}{class_name} {conf:.2f}"
                
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        current_time = time.time()
        if alarm_triggered:
            cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 10)
            cv2.putText(frame, "INTRUSION ALERT!", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            
            if current_time - last_alarm_time > ALARM_COOLDOWN:
                print(f"ALARM TRIGGERED: Detected {detected_pests} at {time.strftime('%H:%M:%S')}.")
                last_alarm_time = current_time

        fps = 1 / (current_time - prev_time) if prev_time else 0
        prev_time = current_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (width - 150, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Agricultural Pest Detection System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()