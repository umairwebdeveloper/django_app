import torch
from shoefitr.models import Settings
from utils.datasets import letterbox
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.general import (
    check_img_size,
    non_max_suppression,
    scale_coords,
)
from utils.depth_calculator import convert
import cv2, numpy as np
import math

device = "cpu"
classes = None
agnostic_nms = False
conf_thres = 0.70
iou_thres = 0.45
augment = False
device = select_device(device)
half = device.type != "cpu"
weights1 = "static/weights/best.pt"
weights2 = "static/weights/best_feet.pt"
device = "cpu"
device = select_device(device)

BALL_MULTIPLIER = 2.8
INSTEP_MULTIPLIER = 3.3
SCALE = 22.87

model1 = attempt_load(weights1, map_location=device)  # load FP32 model
model2 = attempt_load(weights2, map_location=device)  # load FP32 model
names1 = model1.module.names if hasattr(model1, "module") else model1.names
names2 = model2.module.names if hasattr(model2, "module") else model2.names

def fractor_scaled(value,scale):
    return value*scale


def take_first(tup):
    return tup[0]


def get_length(thumb, tiny, sole):
    # mid_point = ((int((thumb[0] + tiny[0]) / 2)), int(((thumb[1] + tiny[1]) / 2) * 0.5))
    foot_length = math.sqrt(
        (math.pow((thumb[0] - sole[0]), 2)) + (math.pow((thumb[1] - sole[1]), 2))
    )
    return thumb, foot_length


def get_ball(thumb, top_right, tiny, top_left):
    ball_point_1 = (
        int((thumb[0] + top_right[0]) / 2),
        int((thumb[1] + top_right[1]) / 2),
    )
    ball_point_2 = (int((tiny[0] + top_left[0]) / 2), int((tiny[1] + top_left[1]) / 2))
    foot_ball = BALL_MULTIPLIER * math.sqrt(
        math.pow((ball_point_1[0] - ball_point_2[0]), 2)
        + math.pow((ball_point_1[1] - ball_point_2[1]), 2)
    )
    return ball_point_1, ball_point_2, foot_ball


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
    foot_waist = (
        math.sqrt(
            (math.pow((top_left[0] - top_right[0]), 2))
            + (math.pow((top_left[1] - top_right[1]), 2))
        )
    ) * 1.1
    return draw_points, foot_waist


def get_instep(top_left, bottom_left, top_right, bottom_right):
    foot_instep = (
        math.sqrt(
            (math.pow((bottom_left[0] - bottom_right[0]), 2))
            + (math.pow((bottom_left[1] - bottom_right[1]), 2))
        )
    ) * INSTEP_MULTIPLIER

    mid_point_1_a = (
        (top_left[0] + bottom_left[0]) / 2,
        (top_left[1] + bottom_left[1]) / 2,
    )
    mid_point_1_b = (
        (top_right[0] + bottom_right[0]) / 2,
        (top_right[1] + bottom_right[1]) / 2,
    )

    mid_point_2_a = (
        (bottom_left[0] + mid_point_1_a[0]) / 2,
        (bottom_left[1] + mid_point_1_a[1]) / 2,
    )
    mid_point_2_b = (
        (bottom_right[0] + mid_point_1_b[0]) / 2,
        (bottom_right[1] + mid_point_1_b[1]) / 2,
    )
    mid_point = (
        (mid_point_2_a[0] + mid_point_2_b[0]) / 2,
        (mid_point_2_a[1] + mid_point_2_b[1]) / 2,
    )

    x_points = np.array([bottom_left[0], mid_point[0], bottom_right[0]])
    y_points = np.array([bottom_left[1], mid_point[1], bottom_right[1]])

    line_space_x = np.linspace(bottom_right[0], bottom_left[0])
    draw_x = line_space_x

    z = np.polyfit(x_points, y_points, 2)
    draw_y = np.polyval(z, draw_x)
    draw_points = np.array([draw_x, draw_y]).T.astype(np.int32)
    return draw_points, foot_instep


def detect(img0, model, imgsz, names):
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    if device.type != "cpu":
        model(
            torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters()))
        )  # run once

    img = letterbox(img0, 640, stride=stride)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img, augment=augment)[0]

    pred = non_max_suppression(
        pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms
    )
    lbls = []
    boxes = []
    # Process detections
    for i, det in enumerate(pred):  # detections per image

        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

            for *xyxy, conf, cls in det:
                label = f"{names[int(cls)]} {conf:.2f}"
                # print(label)
                lbl = label.split(" ")[0]
                lbls.append(lbl)
                x1 = int(xyxy[0].item())
                y1 = int(xyxy[1].item())
                x2 = int(xyxy[2].item())
                y2 = int(xyxy[3].item())
                boxes.append((x1, y1, x2, y2))
                if len(boxes) == 2:
                    break
    return lbls, boxes


def detect_inter(img0, model, imgsz, names, foot_index):
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    if device.type != "cpu":
        model(
            torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters()))
        )  # run once

    img = letterbox(img0, imgsz, stride=stride)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img, augment=augment)[0]

    pred = non_max_suppression(
        pred, 0.05, iou_thres, classes=classes, agnostic=agnostic_nms
    )
    if foot_index == 0:
        boxes = {
            "toe": (92, 23),
            "up1": (105, 82),
            "lw1": (102, 186),
            "sole": (86, 260),
            "lw2": (30, 170),
            "up2": (22, 111),
            "tiny": (28, 68),
        }
    else:
        boxes = {
            "toe": (28, 26),
            "up1": (21, 87),
            "lw1": (25, 181),
            "sole": (35, 255),
            "lw2": (94, 178),
            "up2": (104, 103),
            "tiny": (96, 60),
        }

    # Process detections
    toe_flag = False
    up1_flag = False
    lw1_flag = False
    sole_flag = False
    lw2_flag = False
    up2_flag = False
    tiny_flag = False
    draw_img = img0.copy()
    for i, det in enumerate(pred):  # detections per image

        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

            for *xyxy, conf, cls in det:
                label = f"{names[int(cls)]} {conf:.2f}"
                lbl = label.split(" ")[0]

                x1 = int(xyxy[0].item())
                y1 = int(xyxy[1].item())
                x2 = int(xyxy[2].item())
                y2 = int(xyxy[3].item())
                if lbl == "toe" and toe_flag == False:
                    cv2.rectangle(
                        draw_img, (x1, y1), (x2, y2), color=[255, 0, 0], thickness=1
                    )
                    boxes["toe"] = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    toe_flag = True

                elif lbl == "up1" and up1_flag == False:
                    cv2.rectangle(
                        draw_img, (x1, y1), (x2, y2), color=[255, 0, 0], thickness=1
                    )

                    boxes["up1"] = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    up1_flag = True

                elif lbl == "lw1" and lw1_flag == False:
                    cv2.rectangle(
                        draw_img, (x1, y1), (x2, y2), color=[255, 0, 0], thickness=1
                    )

                    boxes["lw1"] = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    lw1_flag = True

                elif lbl == "sole" and sole_flag == False:
                    cv2.rectangle(
                        draw_img, (x1, y1), (x2, y2), color=[255, 0, 0], thickness=1
                    )

                    boxes["sole"] = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    sole_flag = True

                elif lbl == "up2" and up2_flag == False:
                    cv2.rectangle(
                        draw_img, (x1, y1), (x2, y2), color=[255, 0, 0], thickness=1
                    )

                    boxes["up2"] = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    up2_flag = True

                elif lbl == "lw2" and lw2_flag == False:
                    cv2.rectangle(
                        draw_img, (x1, y1), (x2, y2), color=[255, 0, 0], thickness=1
                    )

                    boxes["lw2"] = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    lw2_flag = True

                elif lbl == "tiny" and tiny_flag == False:
                    cv2.rectangle(
                        draw_img, (x1, y1), (x2, y2), color=[255, 0, 0], thickness=1
                    )
                    boxes["tiny"] = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    tiny_flag = True
    return (
        boxes,
        draw_img,
        [toe_flag, up1_flag, lw1_flag, sole_flag, lw2_flag, up2_flag, tiny_flag],
    )


def detect_fun(image, foot_length_inches=-1.0, PPI=-1.0):
    i = 0
    fin_dict = {}
    imgsz = 640
    fresh_img = image.copy()

    labels, boxes = detect(image, model1, imgsz, names1)
    if len(boxes) < 2:
        found_flag = False
        message = "Feet not found, Please capture both feet together"
        return fin_dict, found_flag, message, None

    else:
        boxes = sorted(boxes, key=take_first)
        left_foot_box = boxes[0]
        right_foot_box = boxes[1]

        left_foot_box = [
            left_foot_box[0] - 10 if left_foot_box[0] > 10 else left_foot_box[0],
            left_foot_box[1] - 10 if left_foot_box[1] > 10 else left_foot_box[1],
            left_foot_box[2] + 10,
            left_foot_box[3] + 10,
        ]
        right_foot_box = [
            right_foot_box[0] - 10 if right_foot_box[0] > 10 else right_foot_box[0],
            right_foot_box[1] - 10 if right_foot_box[1] > 10 else right_foot_box[1],
            right_foot_box[2] + 10,
            right_foot_box[3] + 10,
        ]

        crop_for_left = fresh_img[
            left_foot_box[1] - 10
            if left_foot_box[1] > 10
            else left_foot_box[1] : left_foot_box[3] + 10,
            left_foot_box[0] - 10
            if left_foot_box[0] > 10
            else left_foot_box[0] : left_foot_box[2] + 10,
        ]
        crop_for_right = fresh_img[
            right_foot_box[1] - 10
            if right_foot_box[1] > 10
            else right_foot_box[1] : right_foot_box[3] + 10,
            right_foot_box[0] - 10
            if right_foot_box[0] > 10
            else right_foot_box[0] : right_foot_box[2] + 10,
        ]
        crop_foor_left = cv2.resize(crop_for_left, (126, 264))
        crop_foor_right = cv2.resize(crop_for_right, (126, 264))

        h_l, w_l, _ = crop_foor_left.shape
        h_r, w_r, _ = crop_foor_right.shape

        foots = [crop_foor_left, crop_foor_right]
        boxes_ = [left_foot_box, right_foot_box]
        indexes = ["l", "r"]
        ref_length = None
        all_points = []
        all_flags = []
        for i, foot in enumerate(foots):
            frame = foot.copy()
            frame = cv2.resize(frame, (126, 264))

            boxes_inter, draw, data_check = detect_inter(
                frame, model2, imgsz, names2, i
            )
            # cv2.imwrite(f"draw/{b_name}_{i}.jpg", draw)
            foot_points = np.array(
                [
                    boxes_inter["toe"],
                    boxes_inter["up1"],
                    boxes_inter["lw1"],
                    boxes_inter["sole"],
                    boxes_inter["lw2"],
                    boxes_inter["up2"],
                    boxes_inter["tiny"],
                ]
            )

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
                right_foot_points[ii] = np.array([126 - left_point[0], left_point[1]])

        all_feet = [left_foot_points, right_foot_points]

        for i, foot in enumerate(all_feet):

            foot_points = all_feet[i]
            length_point, length = get_length(
                thumb=foot_points[0], tiny=foot_points[-1], sole=foot_points[3]
            )
            if i == 0:
                ref_length = length
            else:
                _scale = ref_length / length
                foot_length_inches = foot_length_inches / _scale
    
            if not PPI > 0:  # if PPI paramater provided
                scale = Settings.objects.first().scale if Settings.objects.first() else SCALE
                PPI = length / foot_length_inches
                mid_y,mid_x = convert(left_foot_box)
                print('mid_x:',mid_x)
                print('mid_y:',mid_y)
                print('Left foot box:',left_foot_box)
                print('length:',length)
                print('scale:',scale)
                # _,predicted = predict(image,midas,mid_y,mid_x)
                # scaled_predicted = fractor_scaled(predicted,scale)
                # print('predicted:',predicted)
                # print('fractor_scaled function:', scaled_predicted)
                # PPI = length * scaled_predicted
                print('PPI:',PPI)


            boll_point_1, boll_point_2, ball = get_ball(
                thumb=foot_points[0],
                top_right=foot_points[1],
                tiny=foot_points[-1],
                top_left=foot_points[-2],
            )
            waist_points, waist = get_waist(
                thumb_top_mid=boll_point_1,
                top_right=foot_points[1],
                tiny_top_mid=boll_point_2,
                top_left=foot_points[-2],
            )
            instep_points, instep = get_instep(
                top_left=foot_points[-2],
                bottom_left=foot_points[-3],
                top_right=foot_points[1],
                bottom_right=foot_points[2],
            )

            fin_dict[f"length_{indexes[i]}"] = int((length / PPI) * 25.6)
            fin_dict[f"ball_{indexes[i]}"] = int((ball / PPI) * 25.6)
            fin_dict[f"waist_{indexes[i]}"] = int((waist / PPI) * 25.6)
            fin_dict[f"instep_{indexes[i]}"] = int((instep / PPI) * 25.6)

            x = boxes_[i][0]
            y = boxes_[i][1]
            w = boxes_[i][2] - boxes_[i][0]
            h = boxes_[i][3] - boxes_[i][1]
            length_point = [
                int(x + (length_point[0] * w / 126)),
                int(y + (length_point[1] * h / 264)),
            ]
            boll_point_1 = (
                int(x + (boll_point_1[0] * w / 126)),
                int(y + (boll_point_1[1] * h / 264)),
            )
            boll_point_2 = (
                int(x + (boll_point_2[0] * w / 126)),
                int(y + (boll_point_2[1] * h / 264)),
            )
            foot_points_new = []
            for f in foot_points:
                foot_points_new.append(
                    [int(x + (f[0] * w / 126)), int(y + (f[1] * h / 264))]
                )
            foot_points = np.array(foot_points_new)

            waist_points_new = []
            for f in waist_points:
                waist_points_new.append(
                    [int(x + (f[0] * w / 126)), int(y + (f[1] * h / 264))]
                )
            waist_points = np.array(waist_points_new)

            instep_points_new = []
            for f in instep_points:
                instep_points_new.append(
                    [int(x + (f[0] * w / 126)), int(y + (f[1] * h / 264))]
                )
            instep_points = np.array(instep_points_new)

            # print(length_point,foot_points[3])
            cv2.arrowedLine(
                fresh_img,
                (int(length_point[0]), int(length_point[1])),
                tuple(foot_points[3]),
                (0, 0, 255),
                2,
            )
            cv2.arrowedLine(
                fresh_img,
                tuple(foot_points[3]),
                (int(length_point[0]), int(length_point[1])),
                (0, 0, 255),
                2,
            )
            cv2.arrowedLine(fresh_img, boll_point_1, boll_point_2, (255, 0, 0), 2)
            cv2.arrowedLine(fresh_img, boll_point_2, boll_point_1, (255, 0, 0), 2)
            cv2.polylines(fresh_img, [waist_points], False, (0, 255, 0), 2)
            cv2.polylines(fresh_img, [instep_points], False, (100, 150, 0), 2)
        # cv2.imwrite(f"static/out_put/{basename}", fresh_img)
        found_flag = True
        message = "Feet found"
        return fin_dict, found_flag, message, fresh_img
