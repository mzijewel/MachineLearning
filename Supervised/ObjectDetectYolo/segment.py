from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("yolov8m-seg.pt")
img = cv2.imread("s.jpeg")

results = model.predict(img, conf=0.5) #
color=(0,0,255) # Color(BGR)

for result in results:
    for mask, box in zip(result.masks.xy, result.boxes):
        points = np.int32([mask])
        # cv2.polylines(img, points, True, (255, 0, 0), 1)
        cv2.fillPoly(img, points, color)

cv2.imshow("Image", img)
cv2.waitKey(0)

# cv2.imwrite("ss.jpg", img) # save image