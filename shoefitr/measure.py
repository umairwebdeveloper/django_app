import argparse
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from .model.experimental import attempt_load
from .utils.datasets import LoadStreams, LoadImages
from .utils.general import (
    check_img_size,
    check_requirements,
    check_imshow,
    non_max_suppression,
    apply_classifier,
    scale_coords,
    xyxy2xywh,
    strip_optimizer,
    set_logging,
    increment_path,
)
from .utils.plots import plot_one_box_new, user_inputs
from .utils.torch_utils import select_device, load_classifier, time_synchronized
import math


def takeSecond(elem):
    return elem[0][0][0]


def detect(source, model, len_size):

    agnostic_nms = False
    augment = False
    classes = None
    conf_thres = 0.1
    device = "cpu"
    img_size = 416
    iou_thres = 0.5
    output = "inference/output"
    save_txt = False
    view_img = False
    out, view_img, save_txt, imgsz = output, view_img, save_txt, img_size
    webcam = False

    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name="resnet101", n=2)  # initialize
        modelc.load_state_dict(
            torch.load("weights/resnet101.pt", map_location=device)["model"]
        ).to(device).eval()

    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, "module") else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    toes = []
    tins = []
    ups = []
    lows = []
    soles = []
    data = None
    image_obtained = None
    for path, img, im0s, vid_cap in dataset:

        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img, augment=augment)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms
        )

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = (
                    path[i],
                    "%g: " % i,
                    im0s[i].copy(),
                    dataset.count,
                )
            else:
                p, s, im0, frame = path, "", im0s, getattr(dataset, "frame", 0)
            hgt, wdt, _ = im0.shape
            s += "%gx%g " % img.shape[2:]  # print string
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape
                ).round()

                # Print results
                # for c in det[:, -1].unique():
                #    n = (det[:, -1] == c).sum()  # detections per class
                #    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in det:

                    if save_img or view_img:  # Add bbox to image
                        label = f"{names[int(cls)]} {conf:.2f}"
                        confidence = float(f"{conf:.2f}")
                        (
                            toes,
                            tins,
                            ups,
                            lows,
                            soles,
                            image_obtained,
                        ) = plot_one_box_new(
                            toes,
                            tins,
                            ups,
                            lows,
                            soles,
                            confidence,
                            xyxy,
                            im0,
                            label=label,
                            color=colors[int(cls)],
                            line_thickness=3,
                        )

            toes.sort(key=takeSecond)
            tins.sort(key=takeSecond)
            ups.sort(key=takeSecond)
            lows.sort(key=takeSecond)
            soles.sort(key=takeSecond)

            h, w, _ = im0.shape

            if len(tins) == 2:
                tins = (tins[0][0], tins[1][0])
            elif len(tins) == 1:
                if (tins[0][0][0][0] + tins[0][0][1][0]) / 2 < w / 2:
                    tins = (tins[0][0], (None, None))
                else:
                    tins = ((None, None), tins[0][0])

            if len(toes) == 2:
                toes = (toes[0][0], toes[1][0])
            elif len(toes) == 1:
                if (toes[0][0][0][0] + toes[0][0][1][0]) / 2 < w / 2:
                    toes = (toes[0][0], (None, None))
                else:
                    toes = ((None, None), toes[0][0])

            if len(ups) == 2:
                ups = (ups[0][0], ups[1][0])

            elif len(ups) == 1:
                if (ups[0][0][0][0] + ups[0][0][1][0]) / 2 < w / 2:
                    ups = (ups[0][0], (None, None))
                else:
                    ups = ((None, None), ups[0][0])

            if len(lows) == 2:
                lows = (lows[0][0], lows[1][0])
            elif len(lows) == 1:
                if (lows[0][0][0][0] + lows[0][0][1][0]) / 2 < w / 2:
                    lows = (lows[0][0], (None, None))
                else:
                    lows = ((None, None), lows[0][0])

            if len(soles) == 2:
                soles = (soles[0][0], soles[1][0])
            elif len(soles) == 1:
                if (soles[0][0][0][0] + soles[0][0][1][0]) / 2 < w / 2:
                    soles = (soles[0][0], (None, None))
                else:
                    soles = ((None, None), soles[0][0])

            # print(toes)
            # print(tins)
            # print(ups)
            # print(lows)
            # print(soles)

            try:
                left_foot = (toes[0], tins[0], ups[0], lows[0], soles[0])
            except:
                print(
                    "Left foot not detected correctly, Results may be a little off"
                )
            try:
                right_foot = (toes[-1], tins[-1], ups[-1], lows[-1], soles[-1])
            except:
                print(
                    "Right foot not detected correctly, Results may be a little off"
                )

            try:
                len_pixels_left = abs(toes[0][0][-1] - soles[0][-1][-1])
            except:
                len_pixels_left = None

            try:
                len_pixels_right = abs(toes[-1][0][-1] - soles[-1][-1][-1])
            except:
                len_pixels_right = None
            # print(1222, "pixels left len", len_pixels_left, hgt / 3)
            # print(1222, "pixels right len", len_pixels_right, hgt / 3)

            if len_pixels_left != None and len_pixels_left < hgt / 3:
                message = "Too Far, Please keep Your Feet little close to camera"
                data = False
                # print(1222, data, message)
                break
            elif len_pixels_right != None and len_pixels_right < hgt / 3:
                message = "Too Far, Please keep Your Feet little close to camera"
                data = False
                # print(1222, data, message)
                break
            elif len_pixels_left != None and len_pixels_left > hgt / 2:
                message = "Too Close, Please keep Your Feet little away from camera"
                data = False
                # print(1222, data, message)
                break
            elif len_pixels_right != None and len_pixels_right > hgt / 2:
                message = "Too Close, Please keep Your Feet little away from camera"
                data = False
                # print(1222, data, message)
                break

            try:
                if len_pixels_left == None:
                    len_pixels_left = len_pixels_right
                elif len_pixels_right == None:
                    len_pixels_right = len_pixels_left
                scale_old = len_pixels_left / len_pixels_right
                if scale_old > 1:
                    scaler = abs(1 - scale_old)
                    scaler = scaler / 2
                    scale = 1 + scaler
                elif scale_old < 1:
                    scaler = abs(1 - scale_old)
                    scaler = scaler / 2
                    scale = 1 - scaler
                else:
                    scale = 0.98
            except:
                scale = 0.98
            # print(scale)
            try:
                try:
                    pixels_per_inch = len_pixels_left // len_size
                except:
                    pixels_per_inch = len_pixels_right // len_size
            except:
                # pixels_per_inch = None
                continue
            try:
                ball = math.sqrt(
                    (toes[0][-1][0] - tins[0][0][0]) ** 2
                    + (-1.5 * (toes[0][0][-1] - tins[0][0][-1])) ** 2
                )
                left_ball_length = round((ball / pixels_per_inch), 2)
            except:
                left_ball_length = None
            try:
                # waist = (math.sqrt(
                # (ups[0][0][0] - ups[0][-1][0]) ** 2 + (2*(toes[0][0][-1] - tins[0][0][-1])) ** 2))
                waist = 1.05 * abs(ups[0][0][0] - ups[0][-1][0])
                left_waist_length = round((waist / pixels_per_inch), 2)
            except:
                left_waist_length = None
            try:
                instep = 1.45 * math.sqrt(
                    (lows[0][0][0] - lows[0][-1][0]) ** 2
                    + (5 * (lows[0][0][-1] - lows[0][-1][-1])) ** 2
                )
                left_instep_length = round((instep / pixels_per_inch), 2)
            except:
                left_instep_length = None

            # try:

            pixels_per_inch = len_pixels_right // len_size
            try:
                ball = math.sqrt(
                    (toes[-1][0][0] - tins[-1][-1][0]) ** 2
                    + (1.5 * (toes[-1][0][-1] - tins[-1][0][-1])) ** 2
                )
                right_ball_length = ball / pixels_per_inch
                if right_ball_length < left_ball_length:
                    right_ball_length = left_ball_length / scale
                else:
                    right_ball_length = left_ball_length * scale
                right_ball_length = round(right_ball_length, 2)

            except:
                right_ball_length = left_ball_length
            try:
                waist = 1.05 * abs(ups[-1][0][0] - ups[-1][-1][0])
                right_waist_length = waist / pixels_per_inch
                if right_waist_length < left_waist_length:
                    right_waist_length = left_waist_length / scale
                else:
                    right_waist_length = left_waist_length * scale

                right_waist_length = round(right_waist_length, 2)
            except:
                right_waist_length = left_waist_length
            try:
                instep = 1.45 * math.sqrt(
                    (lows[-1][0][0] - lows[-1][-1][0]) ** 2
                    + (5 * (lows[-1][0][-1] - lows[-1][-1][-1])) ** 2
                )

                right_instep_length = instep / pixels_per_inch
                if right_instep_length < left_instep_length:
                    right_instep_length = left_instep_length / scale
                else:
                    right_instep_length = left_instep_length * scale
                right_instep_length = round(right_instep_length, 2)
            except:
                right_instep_length = left_instep_length

            if left_ball_length == None:
                left_ball_length = right_ball_length
            if left_waist_length == None:
                left_waist_length = right_waist_length
            if left_instep_length == None:
                left_instep_length = right_instep_length

            if left_ball_length == None:
                left_ball_length = 0
                right_ball_length = 0
            if left_waist_length == None:
                left_waist_length = 0
                right_waist_length = 0
            if left_instep_length == None:
                left_instep_length = 0
                right_instep_length = 0
            print("")
            print("Left Foot Length:", str(int(len_size * 25.4)) + (" mm"))
            # except:
            #     print('no proper detections for right foot...')
        try:
            print("Left Foot Length::", str(int(len_size * 25.4)) + (" mm"))

            print("Left Ball Size:", str(int(left_ball_length * 25.4)) + (" mm"))
            print("Left Waist Size:", str(int(left_waist_length * 25.4)) + (" mm"))
            print(
                "Left Instep Size:", str(int(left_instep_length * 25.4)) + (" mm")
            )

            print("Right Foot Length:", str(int(len_size / scale * 25.4)) + (" mm"))

            print("Right Ball Size:", str(int(right_ball_length * 25.4)) + (" mm"))
            print(
                "Right Waist Size:", str(int(right_waist_length * 25.4)) + (" mm")
            )
            print(
                "Right Instep Size:", str(int(right_instep_length * 25.4)) + (" mm")
            )
            data = {
                "left_foot": str(int(len_size * 25.4)),
                "left_waist": str(int(left_waist_length * 25.4)),
                "left_ball": str(int(left_ball_length * 25.4 * 2.2)),
                "left_instep": str(int(left_instep_length * 25.4 * 1.5)),
                "right_foot": str(int(len_size / scale * 25.4)),
                "right_ball": str(int(right_ball_length * 25.4 * 2.2)),
                "right_waist": str(int(right_waist_length * 25.4)),
                "right_instep": str(int(right_instep_length * 25.4 * 1.5)),
            }
        except:
            if data is not False:
                data = None

    try:
        try:
            left_thumb_in = ((toes[0][0][0] + toes[0][-1][0]) // 2, toes[0][0][-1])
        except:
            left_thumb_in = None
        try:
            left_thumb_ball = (
                (toes[0][-1][0] + ups[0][-1][0]) // 2,
                (toes[0][-1][-1] + ups[0][0][-1]) // 2,
            )
        except:

            left_thumb_ball = None
        try:
            left_tiny_ball = (tins[0][0][0], tins[0][-1][-1])
        except:
            left_tiny_ball = None
        try:
            left_sole = ((toes[0][0][0] + toes[0][-1][0]) // 2, soles[0][-1][-1])
        except:
            left_sole = None
        try:
            right_thumb_in = ((toes[-1][0][0] + toes[-1][-1][0]) // 2, toes[-1][0][-1])
            right_thumb_ball = (
                (toes[-1][0][0] + ups[-1][0][0]) // 2,
                (toes[-1][-1][-1] + ups[-1][0][-1]) // 2,
            )
        except:
            right_thumb_in = None
            right_thumb_ball = None
        try:
            right_tiny_ball = (tins[-1][-1][0], tins[-1][-1][-1])
        except:
            right_tiny_ball = None
        try:
            right_sole = ((toes[-1][0][0] + toes[-1][-1][0]) // 2, soles[-1][-1][-1])
        except:
            right_sole = None
        try:
            left_bound_in = ups[0][0]
            left_bound_out = (ups[0][-1][0], ups[0][0][-1])
        except:
            left_bound_in = None
            left_bound_out = None
        try:
            right_bound_in = ups[-1][0]
            right_bound_out = (ups[-1][-1][0], ups[-1][0][-1])
        except:
            right_bound_in = None
            right_bound_out = None
        try:
            left_lower_in = lows[0][0]
            left_lower_out = (lows[0][-1][0], lows[0][-1][-1])
        except:
            left_lower_in = None
            left_lower_out = None
        try:
            right_lower_out = lows[-1][0]
            right_lower_in = (lows[-1][-1][0], lows[-1][-1][-1])
        except:
            right_lower_out = None
            right_lower_in = None
    except:
        # print(1222, "except")
        left_thumb_in = None
        left_thumb_ball = None
        left_tiny_ball = None
        left_sole = None
        right_thumb_in = None
        right_thumb_ball = None
        right_tiny_ball = None
        right_sole = None
        left_bound_in = None
        left_bound_out = None
        right_bound_in = None
        right_bound_out = None
        left_lower_in = None
        left_lower_out = None
        right_lower_out = None
        right_lower_in = None
    if data is False:
        # print(1222, "data is False", message)
        left_thumb_in = None
        left_thumb_ball = None
        left_tiny_ball = None
        left_sole = None
        right_thumb_in = None
        right_thumb_ball = None
        right_tiny_ball = None
        right_sole = None
        left_bound_in = None
        left_bound_out = None
        right_bound_in = None
        right_bound_out = None
        left_lower_in = None
        left_lower_out = None
        right_lower_out = None
        right_lower_in = None
        return (
            message,
            None,
            left_thumb_in,
            left_thumb_ball,
            left_tiny_ball,
            left_sole,
            right_thumb_in,
            right_thumb_ball,
            right_tiny_ball,
            right_sole,
            left_bound_in,
            left_bound_out,
            right_bound_in,
            right_bound_out,
            left_lower_in,
            left_lower_out,
            right_lower_out,
            right_lower_in,
            image_obtained,
            ups
        )
    message = "success"
    if data is None:
        # print(1222, "data is None", message)

        message = "Feet Not Detected, Try Again!"
        return (
            message,
            data,
            left_thumb_in,
            left_thumb_ball,
            left_tiny_ball,
            left_sole,
            right_thumb_in,
            right_thumb_ball,
            right_tiny_ball,
            right_sole,
            left_bound_in,
            left_bound_out,
            right_bound_in,
            right_bound_out,
            left_lower_in,
            left_lower_out,
            right_lower_out,
            right_lower_in,
            image_obtained,
            ups
        )
    # print(1222, "data is not None", message)
    return (
        message,
        data,
        left_thumb_in,
        left_thumb_ball,
        left_tiny_ball,
        left_sole,
        right_thumb_in,
        right_thumb_ball,
        right_tiny_ball,
        right_sole,
        left_bound_in,
        left_bound_out,
        right_bound_in,
        right_bound_out,
        left_lower_in,
        left_lower_out,
        right_lower_out,
        right_lower_in,
        image_obtained,
        ups
    )


def magic(source, len_size, system, adult):
    weights = "shoefitr/weights/feet_updated.pt"

    # Initialize model
    set_logging()
    device = "cpu"

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model

    # user inputs!!!
    # source = 'shoefitr/feet pic 3.jpg'

    # user inputs
    sure = "y"
    # len_size = 36
    len_size = round(float(len_size), 1)
    if sure == "n":
        rng = "max"
    else:
        rng = None

    # size estimator function
    len_size = user_inputs(len_size, sure, rng, system, adult)

    # detections
    (
        message,
        data,
        left_thumb_in,
        left_thumb_ball,
        left_tiny_ball,
        left_sole,
        right_thumb_in,
        right_thumb_ball,
        right_tiny_ball,
        right_sole,
        left_bound_in,
        left_bound_out,
        right_bound_in,
        right_bound_out,
        lli,
        llo,
        rli,
        rlo,
        image,
        ups
    ) = detect(source.copy(), model, len_size)
    # print(1222, "after detection", message)

    if data is None:
        return message, None, None, None
    # img_ori = cv2.imread(source)
    img_ori = source
    new_width_left_l = left_tiny_ball
    new_width_left_r = (left_thumb_ball[0], left_tiny_ball[1])
    new_width_right_r = right_tiny_ball
    new_width_right_l = (right_thumb_ball[0], right_tiny_ball[1])
    try:
        try:
            if left_thumb_in is not None and left_sole is not None:
                cv2.arrowedLine(img_ori, left_thumb_in, left_sole, (0, 0, 255), 2)
            if right_thumb_in is not None and right_sole is not None:
                cv2.arrowedLine(img_ori, right_thumb_in, right_sole, (0, 0, 255), 2)
            if left_bound_in is not None and left_bound_out is not None:
                cv2.arrowedLine(img_ori, new_width_left_r, new_width_left_l, (255, 0, 0), 2)
            if right_bound_in is not None and right_bound_out is not None:
                cv2.arrowedLine(img_ori, new_width_right_l, new_width_right_r, (255, 0, 0), 2)
        except:
            None
        try:
            if left_sole is not None and left_thumb_in is not None:
                cv2.arrowedLine(img_ori, left_sole, left_thumb_in, (0, 0, 255), 2)
            if right_sole is not None and right_thumb_in is not None:
                cv2.arrowedLine(img_ori, right_sole, right_thumb_in, (0, 0, 255), 2)
            if left_bound_out is not None and left_bound_in is not None:
                cv2.arrowedLine(img_ori, new_width_left_r, new_width_left_l, (255, 0, 0), 2)
            if right_bound_out is not None and right_bound_in is not None:
                cv2.arrowedLine(img_ori, new_width_right_l, new_width_right_r, (255, 0, 0), 2)
        except:
            None
        try:

            linspace_ball_l = np.linspace(new_width_left_r[0], new_width_left_l[0])
            upper_l = ups[0]
            x1 = np.array([new_width_left_r[0], (new_width_left_l[0] + new_width_left_r[0]) // 2, new_width_left_l[0]])
            y1 = np.array([upper_l[0][1], new_width_left_l[1], upper_l[1][1]])
            z1 = np.polyfit(x1, y1, 2)
            draw_x1 = linspace_ball_l
            draw_y1 = np.polyval(z1, draw_x1)
            draw_points = (np.asarray([draw_x1, draw_y1]).T).astype(np.int32)
            cv2.polylines(img_ori, [draw_points], False, (0, 255, 0), 2)
            # cv2.rectangle(img_ori, left_bound_out, left_bound_in, [255,255,255], 2)
            # if left_thumb_ball is not None and left_tiny_ball is not None:
            #     cv2.arrowedLine(img_ori, left_thumb_ball, left_tiny_ball, (0, 255, 0), 2)
            # # if left_thumb_in is not None and left_thumb_ball is not None:
            #     cv2.arrowedLine(img_ori, left_tiny_ball, left_thumb_ball, (0, 255, 0), 2)
            # if right_thumb_ball is not None and right_tiny_ball is not None:
            #     cv2.arrowedLine(img_ori, right_thumb_ball, right_tiny_ball, (0, 255, 0), 2)
            #     cv2.arrowedLine(img_ori, right_tiny_ball, right_thumb_ball, (0, 255, 0), 2)
        except:
            None

        try:
            linspace_ball_r = np.linspace(new_width_right_l[0], new_width_right_r[0])
            upper_r = ups[1]
            x1 = np.array(
                [new_width_right_l[0], (new_width_right_r[0] + new_width_right_l[0]) // 2, new_width_right_r[0]])
            y1 = np.array([upper_r[0][1], new_width_right_r[1], upper_r[1][1]])
            z1 = np.polyfit(x1, y1, 2)
            draw_x1 = linspace_ball_r
            draw_y1 = np.polyval(z1, draw_x1)
            draw_points = (np.asarray([draw_x1, draw_y1]).T).astype(np.int32)
            cv2.polylines(img_ori, [draw_points], False, (0, 255, 0), 2)
        except:
            None

        try:

            linspace1 = np.linspace(llo[0], lli[0])
            x1 = np.array([llo[0], (llo[0] + lli[0]) // 2, lli[0]])
            y1 = np.array([llo[1], llo[1] - abs(llo[1] - lli[1]), llo[1]])
            # left_end_out = (llo[0], llo[1])
            # left_end_in = (lli[0], llo[1])
            # left_p_mid = (llo[0]+lli[0])//2
            # left_p_out = (llo[0]-left_p_mid//4,(llo[1]+lli[1])//2)
            # left_p_in = (3*(llo[0]+lli[0])//4,(llo[1]+lli[1])//2)
            # print(rlo[1], rli[1])
            z1 = np.polyfit(x1, y1, 2)
            draw_x1 = linspace1
            draw_y1 = np.polyval(z1, draw_x1)
            draw_points1 = (np.asarray([draw_x1, draw_y1]).T).astype(np.int32)
            cv2.polylines(img_ori, [draw_points1], False, (100, 150, 0), 2)
        except:
            pass

        try:
            linspace2 = np.linspace(rlo[0], rli[0])
            x2 = np.array([rlo[0], (rlo[0] + rli[0]) // 2, rli[0]])
            y2 = np.array([rlo[1], rlo[1] - abs(rlo[1] - rli[1]), rlo[1]])
            # print(rlo[1], rli[1])
            z2 = np.polyfit(x2, y2, 2)
            draw_x2 = linspace2
            draw_y2 = np.polyval(z2, draw_x2)
            draw_points2 = (np.asarray([draw_x2, draw_y2]).T).astype(np.int32)
            cv2.polylines(img_ori, [draw_points2], False, (100, 150, 0), 2)

        except:
            pass

        return message, data, img_ori, True

    except:
        # print(12222, "except", message)

        print("Please keep Your Feet straight in the frame of the Camera...")
        return message, None, None, None

    # img = cv2.resize(img_ori,(400,600))
    # cv2.imshow('feet', img)
    # cv2.imwrite('result.jpg', img)
    # cv2.waitKey(0)
