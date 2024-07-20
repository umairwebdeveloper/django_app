import torch
import numpy as np
import cv2
from data.augment import LetterBox
from utils.ops import non_max_suppression, scale_boxes
from utils.torch_utils import select_device, smart_inference_mode
from nn.tasks import attempt_load_weights
from utils.checks import check_imgsz
from depth_anything import *
import math
import os
from ultralytics import YOLO

classes = None
agnostic_nms = False
conf_thres = 0.5
iou_thres = 0.45
augment = False
imgsz = 640
device = 'cpu'
weights_v8_1 = 'weights/best_foot.pt'
weights_v8_2 = 'weights/best_v2.pt'
# weights_polygon = 'weights/polygon.pt'
device = select_device(device)
model_1 = attempt_load_weights(weights_v8_1,
                             device=device,
                             inplace=True,
                             fuse=True)
model_2 = attempt_load_weights(weights_v8_2,
                             device=device,
                             inplace=True,
                             fuse=True)
stride_1 = max(int(model_1.stride.max()), 32)  # model stride
stride_2 = max(int(model_2.stride.max()), 32)  # model stride
names_1 = model_1.module.names if hasattr(model_1, 'module') else model_1.names  # get class names
names_2 = model_2.module.names if hasattr(model_2, 'module') else model_2.names  # get class names
# ----model loaded----#
# if not os.path.exists('results'):
#     os.mkdir(path='results')

# if not os.path.exists('results/cropped'):
#     os.mkdir(path='results/cropped')

# if not os.path.exists('results/draw'):
#     os.mkdir(path='results/draw')

# if not os.path.exists('results/output'):
#     os.mkdir(path='results/output')

def take_first(tup):
    return tup[0]

def extract_landmarks(results):
    x,y,w,h = results[0].boxes.xywh[1]
    x,y = int(x.cpu().detach().numpy()),int(y.cpu().detach().numpy())
    return x,y

def extract_landmarks_from_boxes(boxes):
    x,y,w,h = boxes[0]
    return x,y


def get_length(thumb, tiny, sole):
    # mid_point = ((int((thumb[0] + tiny[0]) / 2)), int(((thumb[1] + tiny[1]) / 2) * 0.5))
    foot_length = math.sqrt((math.pow((thumb[0] - sole[0]), 2)) +
                            (math.pow((thumb[1] - sole[1]), 2)))
    return thumb, foot_length


def get_ball(thumb, top_right, tiny, top_left):
    ball_point_1 = (int((thumb[0] + top_right[0]) / 2), int((thumb[1] + top_right[1]) / 2))
    ball_point_2 = (int((tiny[0] + top_left[0]) / 2), int((tiny[1] + top_left[1]) / 2))
    foot_boll = math.sqrt(math.pow((ball_point_1[0] - ball_point_2[0]), 2) +
                          math.pow((ball_point_1[1] - ball_point_2[1]), 2))
    return ball_point_1, ball_point_2, foot_boll


def get_waist(thumb_top_mid, top_right, tiny_top_mid, top_left):
    mid_x_1 = (top_right[0] + top_left[0]) / 2
    mid_x_2 = (thumb_top_mid[0] + tiny_top_mid[0]) / 2
    mid_y_1 = (top_right[1] + top_left[1]) / 2
    mid_y_2 = (thumb_top_mid[0] + tiny_top_mid[0]) / 2
    mid_point = (int((mid_x_1 + mid_x_2) / 2), int((mid_y_1 + mid_y_2) / 2))
    x_points = np.array([top_left[0], mid_point[0], top_right[0]])
    y_points = np.array([top_left[1], mid_point[1], top_right[1]])
    z = np.polyfit(x_points, y_points, 2)
    line_space_x = np.linspace(top_right[0], top_left[0])
    draw_x = line_space_x
    draw_y = np.polyval(z, draw_x)
    draw_points = np.array([draw_x, draw_y]).T.astype(np.int32)
    foot_waist = (math.sqrt((math.pow((top_left[0] - top_right[0]), 2)) +
                            (math.pow((top_left[1] - top_right[1]), 2)))) * 1.05
    return draw_points, foot_waist


def get_instep(top_left, bottom_left, top_right, bottom_right):
    foot_instep = (math.sqrt((math.pow((bottom_left[0] - bottom_right[0]), 2)) +
                             (math.pow((bottom_left[1] - bottom_right[1]), 2)))) * 1.45

    mid_point_1_a = ((top_left[0] + bottom_left[0]) / 2, (top_left[1] + bottom_left[1]) / 2)
    mid_point_1_b = ((top_right[0] + bottom_right[0]) / 2, (top_right[1] + bottom_right[1]) / 2)

    mid_point_2_a = ((bottom_left[0] + mid_point_1_a[0]) / 2, (bottom_left[1] + mid_point_1_a[1]) / 2)
    mid_point_2_b = ((bottom_right[0] + mid_point_1_b[0]) / 2, (bottom_right[1] + mid_point_1_b[1]) / 2)
    mid_point = ((mid_point_2_a[0] + mid_point_2_b[0]) / 2, (mid_point_2_a[1] + mid_point_2_b[1]) / 2)

    x_points = np.array([bottom_left[0], mid_point[0], bottom_right[0]])
    y_points = np.array([bottom_left[1], mid_point[1], bottom_right[1]])

    line_space_x = np.linspace(bottom_right[0], bottom_left[0])
    draw_x = line_space_x

    z = np.polyfit(x_points, y_points, 2)
    draw_y = np.polyval(z, draw_x)
    draw_points = np.array([draw_x, draw_y]).T.astype(np.int32)
    return draw_points, foot_instep

#----checking imagesize----#
def detect_v8(img0, model, imgsz, names,stride):
    imgsz = check_imgsz(imgsz, stride=32, min_dim=1, floor=0)  # check image size

    half = device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()  # to FP16

    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    im = LetterBox(imgsz, stride=stride)(image=img0)  # padded resize
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)  # contiguous
    im = torch.from_numpy(im).to(device)
    im = im.half() if half else im.float()  # uint8 to fp16/32
    im /= 255.0  # 0 - 255 to 0.0 - 1.0
    if im.ndimension() == 3:
        im = im.unsqueeze(0)

    preds = model(im, augment=augment)
    preds = non_max_suppression(preds,
                                    conf_thres,
                                    iou_thres,
                                    agnostic=agnostic_nms
                                    # ,max_det=1000
                                )
    boxes = []
    labels = []
    # print(len(preds))
    for i, det in enumerate(preds):  # detections per image
        # print(i)
        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], img0.shape).round()
        if len(det) > 0:
            for *xyxy, conf, cls in det:
                label = f'{names[int(cls)]} {conf:.2f}'
                # print("label per person", label)
                # print(label)
                lbl = label.split(' ')[0]
                # print("split label", lbl)
                x1 = int(xyxy[0].item())
                y1 = int(xyxy[1].item())
                x2 = int(xyxy[2].item())
                y2 = int(xyxy[3].item())
                boxes.append((x1,y1,x2,y2))
                labels.append(lbl)
    return  boxes, labels


# def detect_v8(img0, model, imgsz, names,stride):
def detect_inter(img0, model, imgsz, names, foot_index,stride):
    imgsz = check_imgsz(imgsz, stride=32, min_dim=1, floor=0)  # check image size

    half = device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()  # to FP16

    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    im = LetterBox(imgsz, stride=stride)(image=img0)  # padded resize
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)  # contiguous
    im = torch.from_numpy(im).to(device)
    im = im.half() if half else im.float()  # uint8 to fp16/32
    im /= 255.0  # 0 - 255 to 0.0 - 1.0
    if im.ndimension() == 3:
        im = im.unsqueeze(0)

    preds = model(im, augment=augment)
    preds = non_max_suppression(preds,
                                    conf_thres,
                                    iou_thres,
                                    agnostic=agnostic_nms
                                    # ,max_det=1000
                                )
    if foot_index == 0:
        boxes = {'toe': (92, 23), 'up1': (105, 82), 'lw1': (102, 186), 'sole': (86, 260), 'lw2': (30, 170),
                 'up2': (22, 111), 'tiny': (28, 68)}
    else:
        boxes = {'toe': (28, 26), 'up1': (21, 87), 'lw1': (25, 181), 'sole': (35, 255), 'lw2': (94, 178),
                 'up2': (104, 103), 'tiny': (96, 60)}

        # Process detections
    toe_flag = False
    up1_flag = False
    lw1_flag = False
    sole_flag = False
    lw2_flag = False
    up2_flag = False
    tiny_flag = False
    draw_img = img0.copy()
    for i, det in enumerate(preds):  # detections per image
        if len(det):
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in det:
                label = f'{names[int(cls)]} {conf:.2f}'
                lbl = label.split(' ')[0]
                x1 = int(xyxy[0].item())
                y1 = int(xyxy[1].item())
                x2 = int(xyxy[2].item())
                y2 = int(xyxy[3].item())
                if lbl == 'toe' and toe_flag == False:
                    cv2.rectangle(draw_img, (x1, y1), (x2, y2), color=[255, 0, 0], thickness=1)
                    boxes['toe'] = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    toe_flag = True
                elif lbl == 'up1' and up1_flag == False:
                    cv2.rectangle(draw_img, (x1, y1), (x2, y2), color=[255, 0, 0], thickness=1)

                    boxes['up1'] = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    up1_flag = True

                elif lbl == 'lw1' and lw1_flag == False:
                    cv2.rectangle(draw_img, (x1, y1), (x2, y2), color=[255, 0, 0], thickness=1)

                    boxes['lw1'] = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    lw1_flag = True

                elif lbl == 'sole' and sole_flag == False:
                    cv2.rectangle(draw_img, (x1, y1), (x2, y2), color=[255, 0, 0], thickness=1)

                    boxes['sole'] = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    sole_flag = True

                elif lbl == 'up2' and up2_flag == False:
                    cv2.rectangle(draw_img, (x1, y1), (x2, y2), color=[255, 0, 0], thickness=1)

                    boxes['up2'] = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    up2_flag = True

                elif lbl == 'lw2' and lw2_flag == False:
                    cv2.rectangle(draw_img, (x1, y1), (x2, y2), color=[255, 0, 0], thickness=1)

                    boxes['lw2'] = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    lw2_flag = True

                elif lbl == 'tiny' and tiny_flag == False:
                    cv2.rectangle(draw_img, (x1, y1), (x2, y2), color=[255, 0, 0], thickness=1)
                    boxes['tiny'] = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    tiny_flag = True
    return boxes, draw_img, [toe_flag, up1_flag, lw1_flag, sole_flag, lw2_flag, up2_flag, tiny_flag]


def detect_fun(image_name):
    size = 1000
    # foot_model = YOLO('best_foot.pt')
    scale = 26.8
    basename = os.path.basename(image_name)
    b_name = basename.split('.')[0]
    try:
        b_name = basename.split(' ')[0]
    except:
        pass
    fin_dict = {}
    imgsz = 640
    if size == 0:
        len_size = 7.85
    elif size == 1.5:
        len_size = 8.12
    elif size == 2:
        len_size = 8.25
    elif size == 2.5:
        len_size = 8.45
    elif size == 3:
        len_size = 8.6
    elif size == 3.5:
        len_size = 8.75
    elif size == 4:
        len_size = 8.85
    elif size == 4.5:
        len_size = 9.12
    elif size == 5:
        len_size = 9.25
    elif size == 5.5:
        len_size = 9.45
    elif size == 6:
        len_size = 9.6
    elif size == 6.5:
        len_size = 9.75
    elif size == 7:
        len_size = 9.85
    elif size == 7.5:
        len_size = 10.12
    elif size == 8:
        len_size = 10.25
    elif size == 8.5:
        len_size = 10.45
    elif size == 9:
        len_size = 10.6
    elif size == 9.5:
        len_size = 10.75
    elif size == 10:
        len_size = 10.85
    elif size == 10.5:
        len_size = 11.12
    elif size == 11:
        len_size = 11.25
    elif size == 11.5:
        len_size = 11.45
    elif size == 12:
        len_size = 11.6
    elif size == 12.5:
        len_size = 11.75
    elif size == 13:
        len_size = 11.8
    elif size == 13.5:
        len_size = 12.12
    elif size == 14:
        len_size = 12.25
    elif size == 14.5:
        len_size = 12.4
    elif size == 15:
        len_size = 12.55

    img_path = image_name

    img0 = cv2.imread(img_path)
    fresh_img = img0.copy()

    boxes, labels = detect_v8(img0, model_1, imgsz, names_1,stride_1)
    if len(boxes) < 2:
        respos_flag = False
        ret = 'Feets not found'
        return fin_dict, respos_flag, ret , ''
    else:
        boxes = sorted(boxes, key=take_first)
        left_foot_box = boxes[0]
        right_foot_box = boxes[1]

        crop_for_left = fresh_img[
                        left_foot_box[1] - 10 if left_foot_box[1] > 10 else left_foot_box[1]:left_foot_box[3] + 10,
                        left_foot_box[0] - 10 if left_foot_box[0] > 10 else left_foot_box[0]:left_foot_box[2] + 10]
        crop_for_right = fresh_img[
                         right_foot_box[1] - 10 if right_foot_box[1] > 10 else right_foot_box[1]:right_foot_box[
                                                                                                     3] + 10,
                         right_foot_box[0] - 10 if right_foot_box[0] > 10 else right_foot_box[0]:right_foot_box[
                                                                                                     2] + 10]
        crop_foor_left = cv2.resize(crop_for_left, (126, 264))
        # cv2.imwrite(f'results/cropped/{b_name}_left.jpg', crop_foor_left)

        # cv2.imwrite("cropped_left", crop_for_left)
        crop_foor_right = cv2.resize(crop_for_right, (126, 264))
        # cv2.imwrite(f'results/cropped/{b_name}_right.jpg', crop_foor_right)
        # cv2.imwrite("crop_foor_right", crop_for_right)
        foot_right_copy =crop_foor_right.copy()
        boxes, labels = detect_v8(foot_right_copy, model_1, imgsz, names_1,stride_1)
        x_dep,y_dep = extract_landmarks_from_boxes(boxes)

        h_l, w_l, _ = crop_foor_left.shape
        h_r, w_r, _ = crop_foor_right.shape

        foots = [crop_foor_left, crop_foor_right]
        boxes_ = [left_foot_box, right_foot_box]
        indexes = ['l', 'r']
        ref_length = None
        all_points = []
        all_flags = []
        for i, foot in enumerate(foots):
            frame = foot.copy()
            frame = cv2.resize(frame, (126, 264))

            boxes_inter, draw, data_check = detect_inter(frame, model_2, imgsz, names_2, i,stride_2)
            # cv2.imwrite(f'results/draw/{b_name}_draw.jpg',draw)
            foot_points = np.array([boxes_inter['toe'], boxes_inter['up1'], boxes_inter['lw1'], boxes_inter['sole']
                                    , boxes_inter['lw2'], boxes_inter['up2'], boxes_inter['tiny']])

            all_points.append(foot_points)
            all_flags.append(data_check)

        left_foot_data = all_flags[0]
        left_foot_points = all_points[0]
        right_foot_data = all_flags[1]
        right_foot_points = all_points[1]

        for ii in range(len(left_foot_data)):
            left_data = left_foot_data[ii]
            left_point = left_foot_points[ii]
            right_data = right_foot_data[ii]
            right_point = right_foot_points[ii]
            if left_data is False and right_data is not False:
                left_foot_points[ii] = np.array([126 - right_point[0], right_point[1]])
            elif right_data is False and left_data is not False:
                right_foot_points[ii] = np.array([126-left_point[0], left_point[1]])

        all_feet = [left_foot_points, right_foot_points]

        for i, foot in enumerate(all_feet):

            foot_points = all_feet[i]
            length_point, length = get_length(thumb=foot_points[0], tiny=foot_points[-1], sole=foot_points[3])
            if i == 0:
                ref_length = length
            else:
                pass
            #     scale = ref_length / length
            #     foot_length_inches = foot_length_inches / scale
            # PPI = length / foot_length_inches
            copy_frame = frame.copy()
            depth = predict_depthany_v2(copy_frame, x=x_dep,y=y_dep)
            # legth_distance_with_depth = length/depth

            # distance_cm_length = legth_distance_with_depth * scale

            boll_point_1, boll_point_2, ball = get_ball(thumb=foot_points[0], top_right=foot_points[1],
                                                        tiny=foot_points[-1],
                                                        top_left=foot_points[-2])
            waist_points, waist = get_waist(thumb_top_mid=boll_point_1, top_right=foot_points[1],
                                            tiny_top_mid=boll_point_2, top_left=foot_points[-2])
            instep_points, instep = get_instep(top_left=foot_points[-2], bottom_left=foot_points[-3],
                                               top_right=foot_points[1], bottom_right=foot_points[2])

            fin_dict[f"length_{indexes[i]}"] = int((length / depth) * scale)
            fin_dict[f"ball_{indexes[i]}"] = int((ball / depth) * scale)
            fin_dict[f"waist_{indexes[i]}"] = int((waist / depth) * scale)
            fin_dict[f"instep_{indexes[i]}"] = int((instep / depth) * scale)

            x = boxes_[i][0]
            y = boxes_[i][1]
            w = boxes_[i][2] - boxes_[i][0]
            h = boxes_[i][3] - boxes_[i][1]
            length_point = [int(x + (length_point[0] * w / 126)), int(y + (length_point[1] * h / 264))]
            boll_point_1 = (int(x + (boll_point_1[0] * w / 126)), int(y + (boll_point_1[1] * h / 264)))
            boll_point_2 = (int(x + (boll_point_2[0] * w / 126)), int(y + (boll_point_2[1] * h / 264)))
            foot_points_new = []
            for f in foot_points:
                foot_points_new.append([int(x + (f[0] * w / 126)), int(y + (f[1] * h / 264))])
            foot_points = np.array(foot_points_new)

            waist_points_new = []
            for f in waist_points:
                waist_points_new.append([int(x + (f[0] * w / 126)), int(y + (f[1] * h / 264))])
            waist_points = np.array(waist_points_new)

            instep_points_new = []
            for f in instep_points:
                instep_points_new.append([int(x + (f[0] * w / 126)), int(y + (f[1] * h / 264))])
            instep_points = np.array(instep_points_new)

            # print(length_point,foot_points[3])
            cv2.arrowedLine(fresh_img, (int(length_point[0]), int(length_point[1])), tuple(foot_points[3]), (0, 0, 255),
                            2)
            cv2.arrowedLine(fresh_img, tuple(foot_points[3]), (int(length_point[0]), int(length_point[1])), (0, 0, 255),
                            2)
            cv2.arrowedLine(fresh_img, boll_point_1, boll_point_2, (255, 0, 0), 2)
            cv2.arrowedLine(fresh_img, boll_point_2, boll_point_1, (255, 0, 0), 2)
            cv2.polylines(fresh_img, [waist_points], False, (0, 255, 0), 2)
            cv2.polylines(fresh_img, [instep_points], False, (100, 150, 0), 2)
        image_path = f'static/out_put/{b_name}_out_img.jpg'
        cv2.imwrite(f'static/out_put/{b_name}_out_img.jpg', fresh_img)
        respos_flag = True
        ret = 'Feet found'
        return fin_dict, respos_flag, ret , image_path


