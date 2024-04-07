import cv2
import os
import time
from ultralytics import YOLO
import torch
import numpy as np

def cap_imgs():
    cap=cv2.VideoCapture(0)
    print('String image capturing....')
    time.sleep(4)
    for c in range(20):
        _,frame=cap.read()
        cv2.imwrite(f'./images/sleep/{c}.jpg',frame)
        cv2.imshow('v',frame)
        time.sleep(2)

        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def realtime_predict():
    model=YOLO('/Users/zahidul/ML/runs/detect/train5/weights/best.pt')
    cap =cv2.VideoCapture(0)
    while cap.isOpened():
        success,frame=cap.read()

        if success:
            results=model(frame)
            annoted_frame=results[0].plot()

            cv2.imshow('y',annoted_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()


def train(): 
    model=YOLO("yolov8s.pt")
    model.train(data='custom.yaml',epochs=10,imgsz=224,plots=True)

    # for annotation https://app.cvat.ai/


    # train using CLI
    # yolo task=detect mode=train model=yolov8s.pt data=data.yaml epochs=2 imgsz=224 plots=True 
    


# model=torch.hub.load('ultralytics/yolov5','custom',path='/Users/zahidul/ML/runs/detect/train3/weights/best.pt',force_reload=True)

def load_model():
    model=YOLO('/Users/zahidul/ML/runs/detect/train5/weights/best.pt')
    # results=model('data/test/images/1.jpeg',show_labels=False,save=True,line_width=1)
    # results=model.predict('data/test/images/1.jpeg')
    results=model(source='data2/images/1.jpg',show=True)
    results[0].save('t.jpg')
    # print(model)


    # using CLI
    # yolo task=detect mode=predict model=/Users/zahidul/ML/runs/detect/train/weights/best.pt conf=0.25 source=data2/images


# model=torch.load('/Users/zahidul/ML/runs/detect/train3/weights/last.pt')
# print(model)
# model=YOLO('yolov8n.pt')

# model.eval()
# print(results[0])

# train()
# load_model()

realtime_predict()



