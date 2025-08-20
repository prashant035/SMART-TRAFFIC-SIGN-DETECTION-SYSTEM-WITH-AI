import cv2
import numpy as np
import os
import threading
import time
import subprocess
from ultralytics import YOLO
from gtts import gTTS


model1 = YOLO('yolov8_trained.pt')  
model2 = YOLO('best.pt')


LABELS_1 = {
    0: 'GREEN LIGHT', 1: 'RED LIGHT', 2: 'Speed Limit 10', 3: 'Speed Limit 100',
    4: 'Speed Limit 110', 5: 'Speed Limit 120', 6: 'Speed Limit 20', 7: 'Speed Limit 30',
    8: 'Speed Limit 40', 9: 'Speed Limit 50', 10: 'Speed Limit 60', 11: 'Speed Limit 70',
    12: 'Speed Limit 80', 13: 'Speed Limit 90', 14: 'Stop no entry'
}

LABELS_2 = {
    0: 'bus stop', 1: 'do not enter', 2: 'do not stop', 3: 'do not turn right',
    4: 'do not turn left', 5: 'do not U-turn', 6: 'enter left lane', 7: 'green light',
    8: 'left-right lane', 9: 'no parking', 10: 'parking', 11: 'pedestrian crossing',
    12: 'pedestrian zebra crossing', 13: 'railway crossing', 14: 'red light',
    15: 'stop', 16: 'T-intersection left', 17: 'traffic light', 18: 'U-turn',
    19: 'warning', 20: 'yellow light'
}   


SPEECH_DIR = os.path.abspath("speech_files")
os.makedirs(SPEECH_DIR, exist_ok=True)


last_spoken_label = None
last_spoken_time = 0
speech_delay = 2 


def generate_speech_files():
    for label in set(LABELS_1.values()).union(set(LABELS_2.values())):
        filename = os.path.join(SPEECH_DIR, f"{label}.mp3")
        if not os.path.exists(filename): 
            tts = gTTS(text=label, lang='en')
            tts.save(filename)
            print(f"🔊 Generated: {filename}")

generate_speech_files() 


def speak(text):
    filename = os.path.join(SPEECH_DIR, f"{text}.mp3")
    if os.path.exists(filename):
        print(f"🎙️ Playing: {text}") 
        subprocess.run(["mpg123", filename], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) 
    else:
        print(f"❌ Error: {filename} not found!")


def apply_nms(detections, threshold=0.5):
    if len(detections) == 0:
        return []
    
    boxes = np.array([d[:4] for d in detections])
    scores = np.array([d[4] for d in detections])

    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 0.5, threshold)

    if indices is None or len(indices) == 0:
        return []
    if isinstance(indices, tuple):
        indices = indices[0] 

    return [detections[i] for i in indices.flatten()]


def process_frame(frame):
    global last_spoken_label, last_spoken_time

    try:
       
        results1 = model1.predict(frame)
        results2 = model2.predict(frame)

        detections = []

        
        for result in results1:
            for box in result.boxes:
                label_index = int(box.cls)
                conf = float(box.conf)
                if conf > 0.5:
                    label_text = LABELS_1.get(label_index, "Unknown")
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append((x1, y1, x2, y2, conf, label_text))

      
        for result in results2:
            for box in result.boxes:
                label_index = int(box.cls)
                conf = float(box.conf)
                if conf > 0.5:
                    label_text = LABELS_2.get(label_index, "Unknown")
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append((x1, y1, x2, y2, conf, label_text))

    
        filtered_detections = apply_nms(detections)

       
        if filtered_detections:
            _, _, _, _, _, current_text = filtered_detections[0]
            current_time = time.time()

            if current_text != last_spoken_label and (current_time - last_spoken_time > speech_delay):
                print(f"🗣️ Speaking: {current_text}")
                tts_thread = threading.Thread(target=speak, args=(current_text,))
                tts_thread.start()
                last_spoken_label = current_text
                last_spoken_time = current_time

       
        for x1, y1, x2, y2, conf, label_text in filtered_detections:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label_text} ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    except Exception as e:
        print(f"❌ Error processing frame: {e}")

    return frame


def main():
    cap = cv2.VideoCapture(0) 

    if not cap.isOpened():
        print("❌ Error: Cannot access the camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Failed to capture frame from the camera.")
            break

        
        frame = process_frame(frame)

      
        cv2.imshow("Traffic Sign Detection", frame)

       
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()