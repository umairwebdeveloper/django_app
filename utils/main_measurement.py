import cv2
import torch
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import numpy as np
from depth_calculator import *
import cv2
import torch
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import shutil
from scipy.interpolate import RectBivariateSpline
# Load a pretrained YOLOv8n model
model = YOLO('best_foot.pt')
midas = torch.hub.load('intel-isl/MiDaS','DPT_Large')
midas.to('cpu')
midas.eval()

#Process image
transforms = torch.hub.load('intel-isl/MiDaS','transforms')
transform = transforms.dpt_transform


# Load YOLOv8 model
#model = YOLO('yolov8n-pose.pt')

# Load video
video_path = "client.mp4"
cap = cv2.VideoCapture(video_path)

def convert(box):
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    return int(x.cpu().detach().numpy()),int(y.cpu().detach().numpy())

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Run inference on 'bus.jpg'
    results = model.predict(frame)  # results list
    #result_keypoint = results[0].keypoints.xy.cpu().numpy()[0]
    # Show the results
    #print("results :",results)
    print("type",type(results[0]))
    print("length",len(results[0]))
    print("boxes1: ",results[0].boxes.xywh[1])
    print("class",results[0].boxes.cls)
    cls = int(results[0].boxes.cls[1].item())
    name = results[0].names[cls]
    #print("keypoint: ",results[0].keypoints)

    print("name :",name)
    x,y,w,h = results[0].boxes.xywh[1]
    x,y = int(x.cpu().detach().numpy()),int(y.cpu().detach().numpy())

    #image = cv2.circle(frame, (x,y), radius=4, color=(0,255, 255), thickness=-1)
    for r in results:
        frame = r.plot()
    predict(img=frame, midas=midas,mid_x=x,mid_y=y)

#     for r in results:
#         print("r",r)
#         im_array = r.plot()  # plot a BGR numpy array of predictions
#         im_array_rgb = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)

#     # Convert to PIL Image
#         im = Image.fromarray(im_array_rgb)

#     # Display using OpenCV
#         cv2.imshow('Result Image', cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR))
#         im.save('results.jpg')  # save image

#         # # Display the output frame
#         cv2.imshow('YOLOv8 Object Detection', frame)

#         # # Press 'q' to exit the loop
#         # if cv2.waitKey(1) & 0xFF == ord('q'):
#         #     break

# # Release the video capture object and close all windows
#     cv2.waitKey(0)
# cap.release()
# cv2.destroyAllWindows()
