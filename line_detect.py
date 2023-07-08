#The video of the project retrieved from: https://www.pexels.com/video/vehicle-on-highway-with-dash-cam-4608285/
#The codes belongs to Talip Eren Doyan
import cv2
import numpy as np

cap = cv2.VideoCapture("road2.mp4") #Reading video

while True: #Opening endless loop for showing video
    ret,frame = cap.read() #Keeping ret and frame
    if ret == 0: #If ret 0 (if next frame does not exist), finishes the video
        break
    y_start = 240 #Defining y start point for region of interest
    y_end = 640 #Defining y end point for region of interest
    frame = cv2.resize(frame,(640,480)) #Making our frames 640x480
    result = frame.copy() #Copying our frames
    roi = result[y_start:y_end,:] #Defining region of interest. It setted up to see just lines because when it see clouds, it catches clouds also
    gray = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY) #Making frames grayscale
    blur = cv2.GaussianBlur(gray,(5,5),0) #Applying gaussian blur
    edges = cv2.Canny(blur,50,150) #Finding edges of the frame
    _, threshold = cv2.threshold(blur, 200, 250, cv2.THRESH_BINARY) #Making frames black white
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel) #Applying closing to our black-white frame
    lines = cv2.HoughLinesP(opening, 1, np.pi / 180, 50, maxLineGap=30) #Finding hough lines transformation for line detection
    if lines is not None: #If it cannot find line in the frame
        for line in lines:
            (x1, y1, x2, y2) = line[0] #Defining x1,y1,x2,y2 of the lines
            y1 += y_start #y1 added to y_start because y_start is our roi
            y2 += y_start #y2 added to y_start because y_start is our roi
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5) #Drawing green lines to lines
    else:
        cv2.putText(result, "No lines detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.imshow("Frame",frame) #Showing frames to user
    if cv2.waitKey(10) & 0xFF == ord("q"): #If user pushes q it finishes the video
        break

cap.release() #Cap release
cv2.destroyAllWindows()

#Thank you for viewing my codes, please do not hesitate to reach me for correcting, asking, suggesting etc at doyaneren@gmail.com!