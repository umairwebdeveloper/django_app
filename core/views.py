from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from .models import Video
from .serializers import VideoSerializer, FileUploadSerializer
from rest_framework.parsers import MultiPartParser, FormParser
import cv2
import math
import imutils
from ultralytics import YOLO
import torch
import numpy as np
from .calculate_size.depth_anything import *
model = YOLO('core/calculate_size/best_v2.pt')
foot_model = YOLO('core/calculate_size/best_foot.pt')

class VideoUploadView(APIView):
    def post(self, request, *args, **kwargs):
        file_serializer = VideoSerializer(data=request.data)
        if file_serializer.is_valid():
            file_serializer.save()
            return Response(file_serializer.data, status=status.HTTP_201_CREATED)
        else:
            return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class TestApi(APIView):
    def get(self, request, format=None):
        return Response({"name": "hello world"})
    
def extract_landmarks(results):
    x,y,w,h = results[0].boxes.xywh[1]
    x,y = int(x.cpu().detach().numpy()),int(y.cpu().detach().numpy())
    return x,y
    # midas = torch.hub.load('intel-isl/MiDaS','DPT_Large')
    # midas.to('cpu')
    # midas.eval()

    # #Process image
    # transforms = torch.hub.load('intel-isl/MiDaS','transforms')
    # transform = transforms.dpt_transform


    # Load YOLOv8 model
    #model = YOLO('yolov8n-pose.pt')

    # Load video
def calculate_average(numbers):
    if len(numbers) == 0:
        return 0
    return sum(numbers) / len(numbers)

def limited_func_lower(x, y):
    if x >= y:
        return x
    elif x < y:
        return y + (x / 10)

def limited_func(x, y):
    if x <= y:
        return x
    elif x > y:
        return y + (x - y) / 10
    
def cm_to_mm(cm):
    return cm*10
    

def convert(box):
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    return int(x.cpu().detach().numpy()),int(y.cpu().detach().numpy())
    
class CalculateSizeView(APIView):
    parser_classes = (MultiPartParser, FormParser)
    
    def post(self, request, *args, **kwargs):
        serializer = VideoSerializer(data=request.data)
        if serializer.is_valid():
            video_instance = serializer.save()
            file = video_instance.video
            video_path = file.path
            dpi = 33 # Change this value based on your image's DPI
            cap = cv2.VideoCapture(video_path)
            scale = 9.4
            total_depth = 0
            sample_count = 0
            lengths = []
            anydepth = []
            midass = []
            total_length_distance = 0
            length_meature_count = 0
            total_width_distance = 0
            width_measure_count = 0
            frame_counter = 0
            average_length_cm =-1000
            average_width_cm =- 1000
            height_count = 0
            width_count = 0
            sample_limit = 5
            lengths_cm =[]
            width_cm = []
            while cap.isOpened():
                toes_sole = []
                up1_up2 = []
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_counter % 8 != 0:
                    continue

                frame = imutils.resize(frame, height=848)

                original_height, original_width = frame.shape[:2]
                print(original_height, original_width)

                # Calculate the new dimensions
                new_width = original_width
                new_height = original_height 

                # Resize the image
                results = foot_model.predict(frame)
                try:
                    x_dep,y_dep = extract_landmarks(results)
                except:
                    return Response({"error": "error in video"}, status=status.HTTP_400_BAD_REQUEST)
                copy_frame = frame.copy()
                # anydepth.append(depth)

                # depth = predict(img=frame,midas=midas, mid_x=x,mid_y=y)
                #     total_depth += depth
                #     sample_count += 1
                #     average_depth = total_depth / sample_count
                #     print("midas_depth :",depth)
                #     print("midas_avg depth :",average_depth)
                #     midass.append(depth)
                #     df = pd.DataFrame({
                #     'midas': midass,
                #     'anydepth': anydepth
                # })
                # df.to_csv("depth_data.csv")
                # frame = cv2.resize(frame, (new_width, new_height), interpolation = cv2.INTER_AREA)
                # Run inference on 'bus.jpg'
                results = model.predict(frame,conf=0.05)  # results list
                toes_box = []
                sole_box = []
                up1_box =[]
                up2_box = []
                clsses = results[0].boxes.cls
                for box in results[0].boxes:
                    class_id = int(box.cls) 
                    if class_id==0:
                        toes_box.append(box.xywh)
                    if class_id==3:
                        sole_box.append(box.xywh)
                    if class_id==1:
                        up1_box.append(box.xywh)
                    if class_id==5:
                        up2_box.append(box.xywh)

                if (len(toes_box)>0 and len(sole_box)>0):
                    height_count = height_count+ 1

                if (len(up1_box)>0 and len(up2_box)>0):
                    width_count = width_count + 1

                if (len(toes_box)>0 and len(sole_box)>0) or (len(up1_box)>0 and len(up2_box)>0):
                    depth = predict_depthany_v2(copy_frame, x=x_dep,y=y_dep)
                    total_depth += depth
                    sample_count += 1
                    average_depth = total_depth / sample_count
                    print("depth :",depth)
                    print("foot depth :",average_depth)

                else:
                    pass


                # toes_sole = []
                max_dist = 1000
                if len(up1_box)>0 and len(up2_box) >0:
                    for up1 in up1_box:
                        up1x,up1y,up1w,up1h = up1[0]
                        up1x,up1y = int(up1x.cpu().detach().numpy()),int(up1y.cpu().detach().numpy())
                        for up2 in up2_box:
                            up2x,up2y,up2w,up2h = up2[0]
                            up2x,up2y = int(up2x.cpu().detach().numpy()),int(up2y.cpu().detach().numpy())
                            up_dist = np.abs(up1x - up2x)
                            if up_dist<=max_dist:
                                max_dist = up_dist
                                # print(tx,ty,sx,sy)
                                up1_up2 = [up2x,up2y,up1x,up1y]
                    if len(up1_up2)>0:
                        up2x,up2y,up1x,up1y =  up1_up2
                        width_point1 = (up2x,up2y)
                        width_point2 = (up1x,up1y)
                        frame = cv2.circle(frame, width_point1, radius=5, color =(255,255,0), thickness =-1)
                        frame = cv2.circle(frame, width_point2, radius=5, color=(255,255, 0), thickness=-1)
                        cv2.line(frame, width_point1, width_point2, (255, 0, 0), 2)
                        width_distance = math.sqrt((width_point2[0] - width_point1[0])**2 + (width_point2[1] - width_point1[1])**2)
                        width_distance_inches = width_distance / depth

                        # Convert inches to centimeters
                        distance_cm_width = width_distance_inches * scale
                        width_cm.append(distance_cm_width)
                        total_width_distance += distance_cm_width
                        width_measure_count += 1
                        average_width = total_width_distance / width_measure_count

                        # Display the distance on the frame
                        distance_text_width = f"Width of Foot: {distance_cm_width:.2f} cm"
                        # cv2.putText(frame, distance_text_width, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                if len(sole_box)>0 and len(toes_box) >0:
                    for sole in sole_box:
                        sx,sy,sw,sh = sole[0]
                        sx,sy = int(sx.cpu().detach().numpy()),int(sy.cpu().detach().numpy())
                        for toe in toes_box:
                            tx,ty,tw,th = toe[0]
                            tx,ty = int(tx.cpu().detach().numpy()),int(ty.cpu().detach().numpy())
                            dist = np.abs(sx - tx)
                            if dist<=max_dist:
                                max_dist = dist
                                # print(tx,ty,sx,sy)
                                toes_sole = [tx,ty,sx,sy]
                    if len(toes_sole)>0:
                        tx,ty,sx,sy =  toes_sole
                        point1 = (tx,ty)
                        point2 =(sx,sy)
                        frame = cv2.circle(frame, point1, radius=5, color =(0,255,0), thickness =-1)
                        frame = cv2.circle(frame, point2, radius=5, color=(0,255, 0), thickness=-1)
                        cv2.line(frame, point1, point2, (255, 0, 0), 2)
                        legth_distance = math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
                        legth_distance_inches = legth_distance / depth

                        # Convert inches to centimeters
                        distance_cm_length = legth_distance_inches * scale
                        lengths_cm.append(distance_cm_length)
                        total_length_distance += distance_cm_length
                        length_meature_count += 1
                        average_length = total_length_distance / length_meature_count

                        # Display the distance on the frame
                        # print("legth cm :",distance_cm_length)
                        # lengths.append(distance_cm_length)
                        # df = pd.DataFrame({
                        #     'length_cm': lengths,
                        # })
                        # df.to_csv("legth_data.csv")

                if height_count > sample_limit and width_count > sample_limit:
                    break
            distance_text_lenght = f"Legth of Foot: {distance_cm_length:.2f} cm"
            # cv2.putText(frame, distance_text_lenght, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            # cv2.imwrite("foot_size.png", frame)
            # cv2.imshow('foot  measurement',frame)
            # cv2.destroyAllWindows()
            # average_length_cm = (average_length /average_depth)*scale
            # average_width_cm = (average_width /average_depth)*scale
            average_length = calculate_average(lengths_cm)
            average_width = calculate_average(width_cm)
            average_length_mm = cm_to_mm(average_length)
            average_width_mm = cm_to_mm(average_width)
            average_length_mm = limited_func_lower(x = average_length_mm,y=190)
            average_length_mm = limited_func(x =average_length_mm,y= 270)
            
            size_text = f"Legth and width of Foot: {average_length_mm:.2f} {average_width_mm:.2f} mm"
            # cv2.putText(frame, size_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            # cv2.imshow('foot feature keypoint Detection', frame)
            # cv2.waitKey(0)
            # cap.release()
            # cv2.destroyAllWindows()
            # Perform any additional processing with the file contents
            return Response({average_length_mm, average_width_mm}, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
