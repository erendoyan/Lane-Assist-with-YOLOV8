#The video of the project retrieved from: https://www.pexels.com/video/vehicle-on-highway-with-dash-cam-4608285/
#The codes belongs to Talip Eren Doyan
import cv2
import numpy as np
from ultralytics import YOLO

fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Codec for MP4 video
output_fps = 30  # Output frames per second
output_size = (640, 480)  # Output frame size
out = cv2.VideoWriter("video3.mp4",fourcc,output_fps,output_size) #VideoWriter object for saving video
font = cv2.FONT_HERSHEY_SIMPLEX
model = YOLO("yolov8m.pt") #Downloading YoloV8 Medium model (You can also choose x or s model)

#Defining labels
labels=[ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush' ]
cap = cv2.VideoCapture("road2.mp4") #Reading video

while True: #Opening endless loop for showing video
    ret,frame = cap.read() #Keeping ret and frame
    if ret == 0: #If ret 0 (if next frame does not exist), finishes the video
        break
    x_start = 50
    y_start = 300 #Defining y start point for region of interest
    y_end = 640 #Defining y end point for region of interest
    frame = cv2.resize(frame,(640,480)) #Making our frames 640x480
    result = frame.copy() #Copying our frames
    roi = result[y_start:y_end,x_start:] #Defining region of interest. It setted up to see just lines because when it see clouds, it catches clouds also
    gray = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY) #Making frames grayscale
    blur = cv2.GaussianBlur(gray,(5,5),0) #Applying gaussian blur
    edges = cv2.Canny(blur,50,150) #Finding edges of the frame
    _, threshold = cv2.threshold(blur, 190, 250, cv2.THRESH_BINARY) #Making frames black white
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel) #Applying closing to our black-white frame
    lines = cv2.HoughLinesP(opening, 1, np.pi / 180, 50, maxLineGap=30) #Finding hough lines transformation for line detection
    if lines is not None: #If it cannot find line in the frame
        for line in lines:
            (x1, y1, x2, y2) = line[0] #Defining x1,y1,x2,y2 of the lines
            cv2.line(roi, (x1, y1), (x2, y2), (0, 255, 0), 5) #Drawing green lines to lines
    else:
        cv2.putText(result, "No lines detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    frame[y_start:y_end,x_start:] = roi
    img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) #Changing frames format
    results = model(img, verbose=False) #Creating model

    for i in range(len(results[0].boxes)):
        x1, y1, x2, y2 = results[0].boxes.xyxy[i] #Having edge points of the rectangle
        score = results[0].boxes.conf[i] #Score of the predictions
        label = results[0].boxes.cls[i] #Labels of the predictions
        x1, y1, x2, y2, score, label = int(x1), int(y1), int(x2), int(y2), float(score), int(label) #Changing format of the variables
        name = labels[label] #Having name of the labels
        if score < 0.5: #If score <0.5 do not show anything
            continue

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2) #Drawing rectangle
        text = name + ' '
        cv2.putText(frame, text, (x1, y1 - 10), font, 1.2, (255, 0, 255), 2) #Putting text to screen
    out.write(frame) #Saving the video
    cv2.imshow("Frame",frame)
    cv2.imshow("Opening",opening)
    if cv2.waitKey(10) & 0xFF == ord("q"): #If user pushes q it finishes the video
        break

cap.release() #Cap release
cv2.destroyAllWindows()

#Thank you for viewing my codes, please do not hesitate to reach me for correcting, asking, suggesting etc at doyaneren@gmail.com!