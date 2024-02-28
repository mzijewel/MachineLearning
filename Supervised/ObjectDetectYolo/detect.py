# ultralytucs : https://github.com/ultralytics/ultralytics



import supervision as sv
from ultralytics import YOLO
import time
import cv2
import math

# Initialize variables for FPS calculation
start_time = time.time()
frame_count = 0

def img_predict(path):
    model=YOLO('yolov8m.pt') # yolov8s.pt is faster but lower probability
    results=model.predict(path,conf=0.2) # 20% min confidence
    # results[0].save('t2.jpg') # save detected image with box

    person=0
    img=results[0].orig_img
    # find conf & class manually
    boxes=results[0].boxes
    for box in boxes:
        conf = math.ceil((box.conf[0] * 100)) / 100 # confidence 
        cls = int(box.cls[0]) # Class Name
        if(cls==0 and conf>0.4): # cls 0 for person
            person+=1

            # Bounding a box
            x1,y1,x2,y2=box.xyxy[0]
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)

    print(f'Total person: {person}')
    
    cv2.imwrite('t.jpg',img) # save edited image
    
    # show image
    # cv2.imshow('img',img)
    # cv2.waitKey(0)
    
    # find conf & class using sv
    # detections = sv.Detections.from_ultralytics(results[0])
    # for i, cls in enumerate(detections.class_id):
    #     conf=detections.confidence[i]
    #     if(cls==0 and conf>0.5):
    #         print(f'cls:{cls} : conf:{conf}')
    #         person+=1
    
    

    

def multi_img_predict(paths):
    model=YOLO('yolov8m.pt') # yolov8s.pt is faster but lower probability
    results=model.predict(paths,conf=0.2) # 20% confidence score
    for idx, r in enumerate(results):
        r.save(f'img{idx}.jpg')
    

def video(path):
    model=YOLO('yolov8s.pt') # model yolov8m is better but a bit slow
    results=model.track(source=path,show=True)
    # results=model.track(source=path,show=True,tracker="bytetrack.yaml")

def cam():
    model=YOLO('yolov8n.pt')
    model.predict(source='0',show=True)

def calculate_fps():
    # Calculate FPS
    end_time = time.time()
    elapsed_time = end_time - start_time
    fps = frame_count / elapsed_time
    cv2.putText()

def image_seg(path):
    model =YOLO('yolov8n-seg.pt')
    results=model.predict(path)
    results[0].save('tt.jpg')

def pose(path):
    model=YOLO("yolov8n-pose.pt")
    results=model.predict(path)

    results[0].save('a.jpg')

# cam()
# img_predict('4.png')
# multi_img_predict(['1.jpg','2.jpg'])
# video('Sample.mp4')
image_seg('s.jpeg')