from ultralytics import YOLO
import cv2
import cvzone
import math
import time

#a_ultra="videos/20230903185021.mp4"
#ops="videos/20230903190343.mp4"  
#ops='/home/jmunoz004/ai_label/videos/20230903180011.mp4'
#ops='/home/jmunoz004/auto_label_project/videos/20230914103308.mp4'
#ops='/home/jmunoz004/auto_label_project/videos/20230903182136.mp4'
#ops='/home/jmunoz004/auto_label_project/videos/20230903182814.mp4'
#ops='/home/jmunoz004/ai_label/videos/20230903204053.mp4'
ops='/home/jmunoz004/ai_label/videos/20230903203833.mp4'
# cap = cv2.VideoCapture(1)  # For Webcam
# cap.set(3, 1280)
# cap.set(4, 720)
cap = cv2.VideoCapture(ops)  # For Video
vid_w, vid_h = int(cap.get(3)), int(cap.get(4))
out = cv2.VideoWriter('model_test_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'),
                      cap.get(cv2.CAP_PROP_FPS), (vid_w, vid_h))


model=YOLO('model_ai.pt')

classNames = ['face mask', 'face shield', 'glasses', 'gloves', 'hairnet', 'hospital bed', 'lights', 'medical instrument', 'monitor', 'scrubs']

prev_frame_time = 0
new_frame_time = 0

while True:
    new_frame_time = time.time()
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2- x1, y2 -y1
            #cvzone.cornerRect(img, (x1, y1, w, h),l=5,rt=2)
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            c_name= classNames[cls]
            if conf >.10:
                if c_name== 'face mask'or c_name== 'face shield'or c_name== 'glasses' or c_name=='hairnet':
                    mycolor= (0,255,0)
                elif c_name=='gloves' or c_name=='scrubs':
                    mycolor=(255,153,255)
                else:
                    mycolor=(255,102,102)
            
               # cvzone.cornerRect(img, (x1, y1, w, h), l=5,t=1)
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(10, y1)), scale=2, thickness=2,offset=3,colorB=mycolor,
                                   colorT=(51,25,0),colorR=mycolor)

    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(fps)

    cv2.imshow("Image", img)
    out.write(img)
    cv2.waitKey(20)


out.release()