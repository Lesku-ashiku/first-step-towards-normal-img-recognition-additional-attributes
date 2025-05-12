import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
from sklearn.cluster import KMeans
 
cap = cv2.VideoCapture("Videos/cars.mp4")  # For Video
 
model = YOLO("models/yolov8l.pt")



 

# mask
mask = cv2.imread("Images/mask.png")
mask = cv2.resize(mask, (1280, 720))
 
# Tracking
tracker = Sort(max_age=60, min_hits=3, iou_threshold=0.3)
limits = [400, 297, 673, 297]

 

#aggiunto da ahiku




def extract_dominant_color(car_crop, k=7):
    
    car_crop = cv2.resize(car_crop, (60, 60)) 
    # Reshape image to a 2D array of pixels
    pixels = car_crop.reshape((-1, 3))

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(pixels)

    # Get the most frequent color (largest cluster)
    dominant_color = kmeans.cluster_centers_[np.argmax(np.bincount(kmeans.labels_))]

    return dominant_color.astype(int)  # Convert to integer values

def classify_car_color(bgr):
    """Classifies car color based on HSV values with improved thresholds."""
    bgr = np.uint8([[bgr]])  # Convert to OpenCV format
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0][0]  # Convert to HSV

    h, s, v = hsv  # Extract hue, saturation, value

    # Neutral colors (White, Gray, Black)
    if v > 200 and s < 50:
        return "White"
    elif v > 50 and v < 200 and s < 50:
        return "Gray"
    elif v < 50:
        return "Black"

    # Color categories based on hue
    if (h <= 10) or (h >= 170 and h <= 180):
        return "Red"
    elif 11 <= h <= 25:
        return "Orange"
    elif 26 <= h <= 35:
        return "Yellow"
    elif 36 <= h <= 85:
        return "Green"
    elif 86 <= h <= 130:
        return "Blue"
    elif 131 <= h <= 160:
        return "Purple"
    else:
        return "Unknown"
    

while True:
    success, img = cap.read()
    if not success: break
    
    imgRegion = cv2.bitwise_and(img, mask)
 
    
    results = model(imgRegion, stream=True, device='cpu')
 
    detections = np.empty((0, 5))
 
    for r in results:
        boxes = r.boxes
        for box in boxes:
            
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1
 
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            print (cls)
            

 
            if cls == 2 and conf > 0.3:
                # cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),
                #                    scale=0.6, thickness=1, offset=3)
                # cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5)
                currentArray = np.array([x1, y1, x2, y2, conf])
                #ahiku 



                car_crop = img[y1:y2, x1:x2]  # Crop car region
                
                if car_crop.size == 0:  # Skip if the region is empty
                    continue
               
                
                dominant_color = extract_dominant_color(car_crop)
                car_color_name = classify_car_color(dominant_color)
                 # Draw bounding box and color label
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, car_color_name, (x1, y1 - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
             continue
 
   
 
    # cvzone.putTextRect(img, f' Count: {len(totalCount)}', (50, 50))
   
 
    cv2.imshow("Image", img)
    # cv2.imshow("ImageRegion", imgRegion)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
