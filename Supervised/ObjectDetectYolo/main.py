# supervision : https://github.com/roboflow/supervision
# ultralytucs : https://github.com/ultralytics/ultralytics


import cv2
import supervision as sv
from ultralytics import YOLO

# object deteciton from image
def image(path):
    model=YOLO('yolov8m.pt') # model yolov8m is better but a bit slow

    img=cv2.imread(path)
    result=model(img)[0]
    detections=sv.Detections.from_ultralytics(result)
    len(detections)


def video(path):
    model=YOLO('yolov8s.pt') # model yolov8m is better but a bit slow
    results=model.track(source=path,show=True)
    # results=model.track(source=path,show=True,tracker="bytetrack.yaml")


image('2.jpg')
# video('v.mp4')