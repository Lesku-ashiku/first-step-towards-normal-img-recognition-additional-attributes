import numpy as np
import pandas as pd
import cv2
import math
from ultralytics import YOLO

# Caricamento modello YOLO
model = YOLO("models/yolov8n.pt")
cap = cv2.VideoCapture("Videos/cars.mp4")

# Lettura del file dei colori
label= 'bana' 


# Funzione per ottenere il colore medio nel bounding box
def get_average_color(img, x1, y1, x2, y2):
    cropped = img[y1:y2, x1:x2]
    if cropped.size == 0:
        return (0, 0, 0)
    avg_color = cv2.mean(cropped)[:3]
    return tuple(map(int, avg_color))  # R, G, B

def classify_rgb_to_basic_color(R, G, B):
    if R > 100 and G < 100 and B < 100:
        return "red"
    elif R < 80 and G < 80 and B > 200:
        return "blue"
    elif R < 80 and G > 180 and B < 80:
        return "green"
    elif R > 200 and G > 200 and B < 100:
        return "yellow"
    elif R > 200 and G > 150 and B < 80:
        return "orange"
    elif R > 180 and G < 100 and B > 180:
        return "pink"
    elif R > 100 and G > 100 and B > 100:
        return "white"
    elif R < 50 and G < 50 and B < 50:
        return "black"
    elif abs(R - G) < 30 and abs(G - B) < 30 and R < 150 :
        return "gray"
    elif R > 100 and B > 100 and G < 100:
        return "purple"
    elif R > 100 and G > 50 and B < 50:
        return "brown"
    else:
        return "other"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    results = model(frame, device='mps', verbose=False)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            if cls == 2 and conf > 0.15:  # solo auto
                avg_b, avg_g, avg_r = get_average_color(frame, x1, y1, x2, y2)
                
            
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"{classify_rgb_to_basic_color(avg_r, avg_g, avg_b)} (R: {avg_r}, G: {avg_g}, B: {avg_b})", 
                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Auto Rosse", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()