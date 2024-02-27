# supervision : https://github.com/roboflow/supervision
# ultralytucs : https://github.com/ultralytics/ultralytics


import supervision as sv
from ultralytics import YOLO


def img_predict(path):
    model=YOLO('yolov8m.pt') # yolov8s.pt is faster but lower probability
    results=model.predict(path,conf=0.2) # 20% min confidence
    results[0].save('t2.jpg') # save detected image with box
    result=results[0]
    detections = sv.Detections.from_ultralytics(result)
    print(detections)

def multi_img_predict(paths):
    model=YOLO('yolov8m.pt') # yolov8s.pt is faster but lower probability
    results=model.predict(paths,conf=0.2) # 20% confidence score
    for idx, r in enumerate(results):
        r.save(f'img{idx}.jpg')
    

def video(path):
    model=YOLO('yolov8s.pt') # model yolov8m is better but a bit slow
    results=model.track(source=path,show=True)
    # results=model.track(source=path,show=True,tracker="bytetrack.yaml")


img_predict('1.jpg')
# multi_img_predict(['1.jpg','2.jpg'])
# video('v.mp4')