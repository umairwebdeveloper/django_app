import base64

import cv2
import pandas as pd
from django.core.files.base import ContentFile
from numpy import random
from shoefitr1.models import Shoes, Shop, User, data, margin, Reference
import numpy as np
from statistics import mean


def detect_square(image):
    ## convert to hsv
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_green = np.array([42, 40, 40], dtype="uint8")

    upper_green = np.array([70, 255, 255], dtype="uint8")

    mask = cv2.inRange(hsv, lower_green, upper_green)
    detected_output = cv2.bitwise_and(image, image, mask=mask)

    # cv2.imshow("green color detection", mask)

    # median blur
    median = cv2.medianBlur(mask, 3)

    # cv2.imshow("median", median)

    # sharpening the image
    # kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    # sharpened = cv2.filter2D(median, -1, kernel)

    # apply morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # contour detection

    contours, hierarchy = cv2.findContours(
        clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # find contour with biggest area
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    print(area)
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(clean, [box], 0, (255, 0, 0), 1)

    epsilon = 0.03 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, epsilon, True)
    print(approx)

    mid_y_1 = mean((approx[0][0][1], approx[1][0][1]))
    mid_y_2 = mean((approx[2][0][1], approx[3][0][1]))
    mid_x_1 = mean((approx[0][0][0], approx[1][0][0]))
    mid_x_2 = mean((approx[2][0][0], approx[3][0][0]))

    # draw a line from the middle of the first side to the middle of the second side
    def distanceCalculate(p1, p2):
        """p1 and p2 in format (x1,y1) and (x2,y2) tuples"""
        dis = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
        return int(dis)

    # Devide distance with 15 to get PPI
    PPI = distanceCalculate((mid_x_1, mid_y_1), (mid_x_2, mid_y_2)) / 15
    return PPI
    cv2.line(image, (mid_x_1, mid_y_1), (mid_x_2, mid_y_2), (0, 255, 0), 2)

    # draw a line

    cv2.drawContours(clean, [approx], 0, (0, 0, 255), 1)

    print(box)
    # cv2.drawContours(image, contours, -1, (0, 0, 255), 2)
    # cv2.imshow("contours", image)

    # cv2.imshow("sharpened", clean)

    # cv2.waitKey(0)


def user_inputs(size, region, selection, sure=True, rng="max"):
    len_size = 0

    if selection == "adult":
        region_list = ["eur", "uk", "us_m", "us_m_atl", "us_w_fia", "us_w_atl", "jp"]

        if region == "EU":
            size_list = [
                32,
                33,
                34,
                35,
                36,
                37,
                38,
                39,
                40,
                41,
                42,
                43,
                44,
                45,
                46,
                47,
                48,
                49,
            ]
            if size == 32:
                len_size = 8
            elif size == 33:
                len_size = 8.26
            elif size == 34:
                len_size = 8.52
            elif size == 35:
                len_size = 8.78
            elif size == 36:
                len_size = 9.04
            elif size == 37:
                len_size = 9.3
            elif size == 38:
                len_size = 9.56
            elif size == 39:
                len_size = 9.82
            elif size == 40:
                len_size = 10.08
            elif size == 41:
                len_size = 10.34
            elif size == 42:
                len_size = 10.6
            elif size == 43:
                len_size = 10.86
            elif size == 44:
                len_size = 11.12
            elif size == 45:
                len_size = 11.38
            elif size == 46:
                len_size = 11.64
            elif size == 47:
                len_size = 11.9
            elif size == 48:
                len_size = 12.16
            elif size == 49:
                len_size = 12.42
            elif size == 50:
                len_size = 12.68
            elif size == 50.5:
                len_size = 12.94
            elif size == 51:
                len_size = 13.2
            elif size == 51.5:
                len_size = 13.46
            elif size == 52:
                len_size = 13.72

        elif region == "UK":
            size_list = [
                0,
                0.5,
                1,
                1.5,
                2,
                2.5,
                3,
                3.5,
                4,
                4.5,
                5.5,
                6,
                6.5,
                7,
                7.5,
                8,
                8.5,
                9,
                9.5,
                10,
                10.5,
                11,
                11.5,
                12,
                12.5,
                13,
                13.5,
                14,
            ]
            if size == 0:
                len_size = 7.85
            elif size == 0.5:
                len_size = 8.02
            elif size == 1:
                len_size = 8.18
            elif size == 1.5:
                len_size = 8.35
            elif size == 2:
                len_size = 8.51
            elif size == 2.5:
                len_size = 8.68
            elif size == 3:
                len_size = 8.85
            elif size == 3.5:
                len_size = 9.01
            elif size == 4:
                len_size = 9.18
            elif size == 4.5:
                len_size = 9.34
            elif size == 5:
                len_size = 9.51
            elif size == 5.5:
                len_size = 9.68
            elif size == 6:
                len_size = 9.84
            elif size == 6.5:
                len_size = 10.01
            elif size == 7:
                len_size = 10.17
            elif size == 7.5:
                len_size = 10.34
            elif size == 8:
                len_size = 10.51
            elif size == 8.5:
                len_size = 10.67
            elif size == 9:
                len_size = 10.84
            elif size == 9.5:
                len_size = 11.01
            elif size == 10:
                len_size = 11.17
            elif size == 10.5:
                len_size = 11.34
            elif size == 11:
                len_size = 11.51
            elif size == 11.5:
                len_size = 11.67
            elif size == 12:
                len_size = 11.83
            elif size == 12.5:
                len_size = 12
            elif size == 13:
                len_size = 12.17
            elif size == 13.5:
                len_size = 12.33
            elif size == 14:
                len_size = 12.5
            # size_list and if conditions here according to picture
        elif region == "us_m":  # us_m
            size_list = [
                1,
                1.5,
                2,
                2.5,
                3,
                3.5,
                4,
                4.5,
                5,
                5.5,
                6,
                6.5,
                7,
                7.5,
                8,
                8.5,
                9,
                9.5,
                10,
                10.5,
                11,
                11.5,
                12,
                12.5,
                13,
                13.5,
                14,
                14.5,
                15,
            ]
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
            # size_list and if conditions here according to picture

        elif region == "US":  # us_m_atl
            size_list = [
                2,
                2.5,
                3,
                3.5,
                4,
                4.5,
                5.5,
                6,
                6.5,
                7,
                7.5,
                8,
                8.5,
                9,
                9.5,
                10,
                10.5,
                11,
                11.5,
                12,
                12.5,
                13,
                13.5,
                14,
            ]
            if size == 2:
                len_size = 7.87
            elif size == 2.5:
                len_size = 8.08
            elif size == 3:
                len_size = 8.25
            elif size == 3.5:
                len_size = 8.45
            elif size == 4:
                len_size = 8.63
            elif size == 4.5:
                len_size = 8.75
            elif size == 5:
                len_size = 9.05
            elif size == 5.5:
                len_size = 9.25
            elif size == 6:
                len_size = 9.45
            elif size == 6.5:
                len_size = 9.62
            elif size == 7:
                len_size = 9.8
            elif size == 7.5:
                len_size = 10.05
            elif size == 8:
                len_size = 10.22
            elif size == 8.5:
                len_size = 10.45
            elif size == 9:
                len_size = 10.63
            elif size == 9.5:
                len_size = 10.8
            elif size == 10:
                len_size = 11.05
            elif size == 10.5:
                len_size = 11.22
            elif size == 11:
                len_size = 11.4
            elif size == 11.5:
                len_size = 11.61
            elif size == 12:
                len_size = 11.8
            elif size == 12.5:
                len_size = 12
            elif size == 13:
                len_size = 12.2
            elif size == 13.5:
                len_size = 12.39
            elif size == 14:
                len_size = 12.6
            print(9876, "US Man", size, len_size)

            # size_list and if conditions here according to picture

        elif region == "us_w_fia":
            size_list = [
                2,
                2.5,
                3,
                3.5,
                4,
                4.5,
                5,
                5.5,
                6,
                6.5,
                7,
                7.5,
                8,
                8.5,
                9,
                9.5,
                10,
                10.5,
                11,
                11.5,
                12,
                12.5,
                13,
                13.5,
                14,
                14.5,
                15,
                15.5,
                16,
            ]
            if size == 2:
                len_size = 7.85
            elif size == 2.5:
                len_size = 8.12
            elif size == 3:
                len_size = 8.25
            elif size == 3.5:
                len_size = 8.45
            elif size == 4:
                len_size = 8.6
            elif size == 4.5:
                len_size = 8.75
            elif size == 5:
                len_size = 8.85
            elif size == 5.5:
                len_size = 9.12
            elif size == 6:
                len_size = 9.25
            elif size == 6.5:
                len_size = 9.45
            elif size == 7:
                len_size = 9.6
            elif size == 7.5:
                len_size = 9.75
            elif size == 8:
                len_size = 9.85
            elif size == 8.5:
                len_size = 10.12
            elif size == 9:
                len_size = 10.25
            elif size == 9.5:
                len_size = 10.45
            elif size == 10:
                len_size = 10.6
            elif size == 10.5:
                len_size = 10.75
            elif size == 11:
                len_size = 10.85
            elif size == 11.5:
                len_size = 11.12
            elif size == 12:
                len_size = 11.25
            elif size == 12.5:
                len_size = 11.45
            elif size == 13:
                len_size = 11.6
            elif size == 13.5:
                len_size = 11.75
            elif size == 14:
                len_size = 11.8
            elif size == 14.5:
                len_size = 12.12
            elif size == 15:
                len_size = 12.25
            elif size == 15.5:
                len_size = 12.4
            elif size == 16:
                len_size = 12.55
            # size_list and if conditions here according to picture

        elif region == "US_W":  # us_w_atl

            size_list = [
                3,
                3.5,
                4,
                4.5,
                5,
                5.5,
                6,
                6.5,
                7,
                7.5,
                8,
                8.5,
                9,
                9.5,
                10,
                10.5,
                11,
                11.5,
                12,
                12.5,
                13,
                13.5,
                14,
                14.5,
                15,
            ]
            if size == 3:
                len_size = 7.87
            elif size == 3.5:
                len_size = 8.08
            elif size == 4:
                len_size = 8.25
            elif size == 4.5:
                len_size = 8.45
            elif size == 5:
                len_size = 8.63
            elif size == 5.5:
                len_size = 8.75
            elif size == 6:
                len_size = 9.05
            elif size == 6.5:
                len_size = 9.25
            elif size == 7:
                len_size = 9.45
            elif size == 7.5:
                len_size = 9.62
            elif size == 8:
                len_size = 9.8
            elif size == 8.5:
                len_size = 10.05
            elif size == 9:
                len_size = 10.22
            elif size == 9.5:
                len_size = 10.45
            elif size == 10:
                len_size = 10.63
            elif size == 10.5:
                len_size = 10.8
            elif size == 11:
                len_size = 11.05
            elif size == 11.5:
                len_size = 11.22
            elif size == 12:
                len_size = 11.4
            elif size == 12.5:
                len_size = 11.61
            elif size == 13:
                len_size = 11.8
            elif size == 13.5:
                len_size = 12
            elif size == 14:
                len_size = 12.2
            elif size == 14.5:
                len_size = 12.39
            elif size == 15:
                len_size = 12.6
            print(9876, "US Woman", size, len_size)

            # size_list and if conditions here according to picture

        elif region == "JP":
            size_list = [
                20,
                20.5,
                21,
                21.5,
                22,
                22.5,
                23,
                23.5,
                24,
                24.5,
                25,
                25.5,
                26,
                26.5,
                27,
                27.5,
                28,
                28.5,
                29,
                29.5,
                30,
            ]
            if size == 20:
                len_size = 7.87
            elif size == 20.5:
                len_size = 8.08
            elif size == 21:
                len_size = 8.25
            elif size == 21.5:
                len_size = 8.45
            elif size == 22:
                len_size = 8.63
            elif size == 22.5:
                len_size = 8.75
            elif size == 23:
                len_size = 9.05
            elif size == 23.5:
                len_size = 9.25
            elif size == 24:
                len_size = 9.45
            elif size == 24.5:
                len_size = 9.62
            elif size == 25:
                len_size = 9.8
            elif size == 25.5:
                len_size = 10.05
            elif size == 26:
                len_size = 10.22
            elif size == 26.5:
                len_size = 10.45
            elif size == 27:
                len_size = 10.63
            elif size == 27.5:
                len_size = 10.8
            elif size == 28:
                len_size = 11.05
            elif size == 28.5:
                len_size = 11.22
            elif size == 29:
                len_size = 11.4
            elif size == 29.5:
                len_size = 11.61
            elif size == 30:
                len_size = 11.8

            # size_list and if conditions here according to picture

    elif selection == "child":
        region_list = ["eur", "uk", "us", "us_atl", "jp"]

        if region == "EU":
            size_list = [
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
                28,
                29,
                30,
                31,
            ]
            if size == 14:
                len_size = 3.25
            elif size == 15:
                len_size = 3.51
            elif size == 16:
                len_size = 3.77
            elif size == 17:
                len_size = 4.03
            elif size == 18:
                len_size = 4.29
            elif size == 19:
                len_size = 4.55
            elif size == 20:
                len_size = 4.81
            elif size == 21:
                len_size = 5.07
            elif size == 22:
                len_size = 5.33
            elif size == 23:
                len_size = 5.59
            elif size == 24:
                len_size = 5.85
            elif size == 25:
                len_size = 6.11
            elif size == 26:
                len_size = 6.37
            elif size == 27:
                len_size = 6.63
            elif size == 28:
                len_size = 6.89
            elif size == 29:
                len_size = 7.15
            elif size == 30:
                len_size = 7.41
            elif size == 31:
                len_size = 7.67

        elif region == "UK":
            size_list = [
                0,
                0.5,
                1,
                1.5,
                2,
                2.5,
                3,
                3.5,
                4,
                4.5,
                5,
                5.5,
                6,
                6.5,
                7,
                7.5,
                8,
                8.5,
                9,
                9.5,
                10,
                10.5,
                11,
                11.5,
                12,
                12.5,
                13,
            ]
            if size == 0:
                len_size = 3.05
            elif size == 0.5:
                len_size = 3.23
            elif size == 1:
                len_size = 3.9
            elif size == 1.5:
                len_size = 4.07
            elif size == 2:
                len_size = 4.23
            elif size == 2.5:
                len_size = 4.4
            elif size == 3:
                len_size = 4.56
            elif size == 3.5:
                len_size = 4.73
            elif size == 4:
                len_size = 4.9
            elif size == 4.5:
                len_size = 5.06
            elif size == 5:
                len_size = 5.23
            elif size == 5.5:
                len_size = 5.4
            elif size == 6:
                len_size = 5.56
            elif size == 6.5:
                len_size = 5.73
            elif size == 7:
                len_size = 5.9
            elif size == 7.5:
                len_size = 6.06
            elif size == 8:
                len_size = 6.22
            elif size == 8.5:
                len_size = 6.39
            elif size == 9:
                len_size = 6.56
            elif size == 9.5:
                len_size = 6.72
            elif size == 10:
                len_size = 6.89
            elif size == 10.5:
                len_size = 7.05
            elif size == 11:
                len_size = 7.22
            elif size == 11.5:
                len_size = 7.39
            elif size == 12:
                len_size = 7.55
            elif size == 12.5:
                len_size = 7.72
            elif size == 13:
                len_size = 7.88
            # size_list and if conditions here according to picture

        elif region == "us_":
            size_list = [
                1,
                1.5,
                2,
                2.5,
                3,
                3.5,
                4,
                4.5,
                5,
                5.5,
                6,
                6.5,
                7,
                7.5,
                8,
                8.5,
                9,
                9.5,
                10,
                10.5,
                11,
                11.5,
                12,
                12.5,
                13,
            ]
            if size == 1:
                len_size = 3.79
            elif size == 1.5:
                len_size = 3.93
            elif size == 2:
                len_size = 4.12
            elif size == 2.5:
                len_size = 4.31
            elif size == 3:
                len_size = 4.48
            elif size == 3.5:
                len_size = 4.67
            elif size == 4:
                len_size = 4.8
            elif size == 4.5:
                len_size = 4.93
            elif size == 5:
                len_size = 5.1
            elif size == 5.5:
                len_size = 5.17
            elif size == 6:
                len_size = 5.33
            elif size == 6.5:
                len_size = 5.5
            elif size == 7:
                len_size = 5.67
            elif size == 7.5:
                len_size = 5.84
            elif size == 8:
                len_size = 6.11
            elif size == 8.5:
                len_size = 6.28
            elif size == 9:
                len_size = 6.45
            elif size == 9.5:
                len_size = 6.62
            elif size == 10:
                len_size = 6.79
            elif size == 10.5:
                len_size = 6.96
            elif size == 11:
                len_size = 7.13
            elif size == 11.5:
                len_size = 7.3
            elif size == 12:
                len_size = 7.47
            elif size == 12.5:
                len_size = 7.64
            elif size == 13:
                len_size = 7.81
            # size_list and if conditions here according to picture

        elif region == "US":  # us_atl
            size_list = [
                2,
                2.5,
                3,
                3.5,
                4,
                4.5,
                5,
                5.5,
                6,
                6.5,
                7,
                7.5,
                8,
                8.5,
                9,
                9.5,
                10,
                10.5,
                11,
                11.5,
                12,
                12.5,
                13,
                13.5,
            ]
            if size == 2:
                len_size = 3.06
            elif size == 2.5:
                len_size = 3.24
            elif size == 3:
                len_size = 3.44
            elif size == 3.5:
                len_size = 3.64
            elif size == 4:
                len_size = 3.84
            elif size == 4.5:
                len_size = 4.04
            elif size == 5:
                len_size = 4.24
            elif size == 5.5:
                len_size = 4.44
            elif size == 6:
                len_size = 4.64
            elif size == 6.5:
                len_size = 4.84
            elif size == 7:
                len_size = 5.04
            elif size == 7.5:
                len_size = 5.24
            elif size == 8:
                len_size = 5.44
            elif size == 8.5:
                len_size = 5.64
            elif size == 9:
                len_size = 5.84
            elif size == 9.5:
                len_size = 6.04
            elif size == 10:
                len_size = 6.24
            elif size == 10.5:
                len_size = 6.44
            elif size == 11:
                len_size = 6.64
            elif size == 11.5:
                len_size = 6.84
            elif size == 12:
                len_size = 7.04
            elif size == 12.5:
                len_size = 7.24
            elif size == 13:
                len_size = 7.44
            elif size == 13.5:
                len_size = 7.64

            # size_list and if conditions here according to picture

        elif region == "JP":
            size_list = [
                8,
                8.5,
                9,
                9.5,
                10,
                10.5,
                11,
                11.5,
                12,
                12.5,
                13,
                13.5,
                14,
                14.5,
                15,
                15.5,
                16,
                16.5,
                17,
                17.5,
                18,
                18.5,
                19,
                19.5,
                20,
            ]
            if size == 8:
                len_size = 3.12
            elif size == 8.5:
                len_size = 3.31
            elif size == 9:
                len_size = 3.5
            elif size == 9.5:
                len_size = 3.69
            elif size == 10:
                len_size = 3.88
            elif size == 10.5:
                len_size = 4.07
            elif size == 11:
                len_size = 4.26
            elif size == 11.5:
                len_size = 4.45
            elif size == 12:
                len_size = 4.64
            elif size == 12.5:
                len_size = 4.83
            elif size == 13:
                len_size = 5.04
            elif size == 13.5:
                len_size = 5.24
            elif size == 14:
                len_size = 5.44
            elif size == 14.5:
                len_size = 5.64
            elif size == 15:
                len_size = 5.84
            elif size == 15.5:
                len_size = 6.04
            elif size == 16:
                len_size = 6.24
            elif size == 16.5:
                len_size = 6.44
            elif size == 17:
                len_size = 6.64
            elif size == 17.5:
                len_size = 6.84
            elif size == 18:
                len_size = 7.04
            elif size == 18.5:
                len_size = 7.24
            elif size == 19:
                len_size = 7.44
            elif size == 19.5:
                len_size = 7.64
            elif size == 20:
                len_size = 7.84

            # size_list and if conditions here according to picture
    if not sure:
        if rng == "max":
            len_size = len_size - random.uniform((len_size / 50), (len_size / 20))
        if rng == "min":
            len_size = len_size + random.uniform((len_size / 50), (len_size / 20))
    return round(float(len_size), 2)


def to_base64(im):
    _, im_arr = cv2.imencode(".jpg", im)

    im_bytes = im_arr.tobytes()

    encoded_string = base64.b64encode(im_bytes)
    base64_string = encoded_string.decode("UTF-8")
    base64_string = "data:image/jpeg;base64," + base64_string
    return base64_string


def calculate_model_id(
    shopid,
    model_name,
    length,
    width,
    ball,
    instep,
    marginid,
    length_margin=12,
    width_margin=20,
):
    # add checks for some special shops
    if shopid == "demoshop01":
        length_margin = 5
    try:
        # if shopid is not admin then return results
        if shopid != "admin":
            print("margin id", marginid)
            if marginid:
                lum = int(
                    margin.objects.filter(file_data__shop__shopOwner__username=str(shopid))
                    .values("length_up_margin")
                    .order_by("-id")
                    .first()["length_up_margin"]
                    or 0
                )
                ldm = int(
                    margin.objects.filter(file_data__shop__shopOwner__username=str(shopid))
                    .values("length_down_margin")
                    .order_by("-id")
                    .first()["length_down_margin"]
                    or 0
                )
                wum = int(
                    margin.objects.filter(file_data__shop__shopOwner__username=str(shopid))
                    .values("width_up_margin")
                    .order_by("-id")
                    .first()["width_up_margin"]
                    or 0
                )
                wdm = int(
                    margin.objects.filter(file_data__shop__shopOwner__username=str(shopid))
                    .values("width_down_margin")
                    .order_by("-id")
                    .first()["width_down_margin"]
                    or 0
                )

                bum = int(
                    margin.objects.filter(file_data__shop__shopOwner__username=str(shopid))
                    .values("ball_up_margin")
                    .order_by("-id")
                    .first()["ball_up_margin"]
                    or 0
                )
                bdm = int(
                    margin.objects.filter(file_data__shop__shopOwner__username=str(shopid))
                    .values("ball_down_margin")
                    .order_by("-id")
                    .first()["ball_down_margin"]
                    or 0
                )
                ium = int(
                    margin.objects.filter(file_data__shop__shopOwner__username=str(shopid))
                    .values("instep_up_margin")
                    .order_by("-id")
                    .first()["instep_up_margin"]
                    or 0
                )
                idm = int(
                    margin.objects.filter(file_data__shop__shopOwner__username=str(shopid))
                    .values("instep_down_margin")
                    .order_by("-id")
                    .first()["instep_down_margin"]
                    or 0
                )

            else:

                lum = int(
                    data.objects.filter(shop__shopOwner__username=str(shopid))
                    .values("length_up_margin")
                    .order_by("-id")
                    .first()["length_up_margin"]
                    or 0
                )
                ldm = int(
                    data.objects.filter(shop__shopOwner__username=str(shopid))
                    .values("length_down_margin")
                    .order_by("-id")
                    .first()["length_down_margin"]
                    or 0
                )
                wum = int(
                    data.objects.filter(shop__shopOwner__username=str(shopid))
                    .values("width_up_margin")
                    .order_by("-id")
                    .first()["width_up_margin"]
                    or 0
                )
                wdm = int(
                    data.objects.filter(shop__shopOwner__username=str(shopid))
                    .values("width_down_margin")
                    .order_by("-id")
                    .first()["width_down_margin"]
                    or 0
                )

                bum = int(
                    data.objects.filter(shop__shopOwner__username=str(shopid))
                    .values("ball_up_margin")
                    .order_by("-id")
                    .first()["ball_up_margin"]
                    or 0
                )
                bdm = int(
                    data.objects.filter(shop__shopOwner__username=str(shopid))
                    .values("ball_down_margin")
                    .order_by("-id")
                    .first()["ball_down_margin"]
                    or 0
                )
                ium = int(
                    data.objects.filter(shop__shopOwner__username=str(shopid))
                    .values("instep_up_margin")
                    .order_by("-id")
                    .first()["instep_up_margin"]
                    or 0
                )
                idm = int(
                    data.objects.filter(shop__shopOwner__username=str(shopid))
                    .values("instep_down_margin")
                    .order_by("-id")
                    .first()["instep_down_margin"]
                    or 0
                )
            prev_file = (
                data.objects.filter(shop__shopOwner__username=str(shopid)).order_by("-id").first()
            )
            print("File: ", lum, marginid, prev_file.file)
            print("File: ", prev_file)
            if ".xls" in str(prev_file.file):
                df = pd.read_excel(prev_file.file, header=0)
                print("--------xls")
            elif ".csv" in str(prev_file.file):
                try:
                    print("2----------csv")
                    df = pd.read_csv(prev_file.file, delimiter=";", header=0)
                    a = df["Name"]
                except Exception as e:
                    print("3----------csv", e)
                    try:
                        df = pd.read_csv(prev_file.file, delimiter=",", header=0)
                        print(df["Name"])
                    except Exception as e:
                        print(23)
                        print(e)
            try:
                if model_name != "*":
                    df = df.loc[df["Name"] == model_name]
                if df.shape[0] == 0:
                    print(2323, "Wrong Model Name")
                # replace comma with dot
                df = df.replace(",", ".", regex=True)
                df["Length"] = pd.to_numeric(df["Length"], downcast="float")
                df["Width"] = pd.to_numeric(df["Width"], downcast="float")
                try:
                    df["Ball"] = pd.to_numeric(df["Ball"], downcast="float")
                except Exception as e:
                    print("Ball error: 2", e)

                #### Start: Finding filtered row with closest length, width and ball ####

                def length_apply_function(l):
                    if l - length <= lum and length - l <= ldm:
                        return l
                    return 0

                def width_apply_function(w):
                    if w - width <= wum and width - w <= wdm:
                        return 1
                    return 0

                df["Length"] = df["Length"].apply(length_apply_function)
                df["IsWidthFit"] = df["Width"].apply(width_apply_function)

                df_sorted_on_length = df.iloc[(df["Length"] - length).abs().argsort()]
                df_filtered_on_length = df_sorted_on_length[
                    df_sorted_on_length["Length"] > 0
                ]
                try:
                    df_filtered_sorted_on_ball = df_filtered_on_length.iloc[
                        (df_filtered_on_length["Ball"] - ball).abs().argsort()
                    ]
                except:
                    df_filtered_sorted_on_ball = df_filtered_on_length
                df_filtered_sorted_on_width = df_filtered_sorted_on_ball.iloc[
                    (df_filtered_sorted_on_ball["Width"] - width).abs().argsort()
                ]
                sorted_row = df_filtered_sorted_on_width.iloc[0]

                #### End: Finding filtered row with closest length, width and ball ####

                l = int(sorted_row["Length"])
                try:
                    w = int(sorted_row["Width"])
                except Exception as e:
                    # if Width column is not Nan
                    w = None
                    print("Width error:", e)
                try:
                    b = int(sorted_row["Ball"])
                except Exception as e:
                    # if Ball column is not Nan
                    b = None
                    print("Ball error:", e)
                try:
                    i = int(sorted_row["Instep"])
                except Exception as e:
                    # if Instep column is not Nan
                    i = None
                    print("Instep error:", e)
                print(999, length, l, w)
            except Exception as e:
                print(999, e.args[0])
            # Check if length is within up and down margins
            if l - length <= lum and length - l <= ldm:
                size = sorted_row["SizeEU"]
                model_id = sorted_row["ModelId"]
                # if abs(width-w)<=width_margin:

                if w is not None:
                    # Check if width is within up and down margins
                    if w - width <= wum and width - w <= wdm:
                        width_advice = "Fit"
                    elif width - w > wdm:
                        width_advice = "Tight"
                    else:
                        width_advice = "Loose"
                elif w is None:
                    width_advice = "_"

                # if Width column is not available then assume it is fitting
                # and advice on ball
                if w is None and b is not None:
                    width_advice = "Fit"

                if b is not None:
                    if b - ball <= bum and ball - b <= bdm:
                        ball_advice = "Fit"
                    elif ball - b > bdm:
                        ball_advice = "Tight"
                    else:
                        ball_advice = "Loose"
                elif b is None:
                    ball_advice = "_"

                if i is not None:
                    if i - instep <= ium and instep - i <= idm:
                        instep_advice = "Fit"
                    elif instep - i > idm:
                        instep_advice = "Tight"
                    else:
                        instep_advice = "Loose"
                elif i is None:
                    instep_advice = "_"

                print(
                    2,
                    {
                        "size": str(size),
                        "width_advice": width_advice,
                        "ball_advice": ball_advice,
                        "instep_advice": instep_advice,
                        "length": str(l),
                        "width": str(w),
                        "model_id": str(model_id),
                    },
                )
                if w is None:
                    w = 0
                return {
                    "size": str(size),
                    "width_advice": width_advice,
                    "ball_advice": ball_advice,
                    "instep_advice": instep_advice,
                    "length": str(round(float(l), 1)),
                    "width": str(round(float(w), 1)),
                    "model_id": str(model_id),
                }
            # Check if length is not within up and down margins
            else:
                return {
                    "size": "_",
                    "width_advice": "_",
                    "ball_advice": "_",
                    "instep_advice": "_",
                    "model_id": "_",
                    "length": "0",
                    "width": "0",
                }
                print("ERROR: ", 2)

        # if shopid is admin, return nothing
        else:
            return {
                "size": "_",
                "width_advice": "_",
                "ball_advice": "_",
                "instep_advice": "_",
                "model_id": "_",
                "length": "0",
                "width": "0",
            }
    except:
        return {
            "size": "_",
            "width_advice": "_",
            "ball_advice": "_",
            "instep_advice": "_",
            "model_id": "_",
            "length": "0",
            "width": "0",
        }


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def save_to_db(
    shopid,
    userid,
    modelid,
    left_length,
    right_length,
    left_width,
    right_width,
    left_instep,
    right_instep,
    left_ball,
    right_ball,
    size,
    width_advice,
    ball_advice,
    instep_advice,
    model_name,
    size_reference,
    selection,
    region,
    picture,
    shoespair,
):
    confirmation = False
    user_found = User.objects.filter(username=shopid).exclude(username="admin").first()
    if not user_found:
        print(234234, "User not found", shopid)
        return {"saved": confirmation}
    print(234234, picture.shape)
    _, frame_jpg = cv2.imencode(".jpg", picture)
    file = ContentFile(frame_jpg)
    try:
        reference = Reference(size=str(size_reference), region=region, selection=selection)
        reference.save()
        shop = Shop.objects.filter(shopOwner=user_found).first()
        if not shop:
            print(234334, "Shop not found", user_found)
            return {"saved": confirmation}
        instance = Shoes(
            shop=shop,
            userid=userid,
            modelid=modelid,
            left_length=left_length,
            right_length=right_length,
            left_width=left_width,
            right_width=right_width,
            left_instep=left_instep,
            right_instep=right_instep,
            left_ball=left_ball,
            right_ball=right_ball,
            size_eu=size,
            width_advice=width_advice,  # size_uk
            ball_advice=ball_advice,
            instep_advice=instep_advice,
            model_name=model_name,  # size_us
            picture=None,
            reference=reference,
            shoespair=shoespair,
        )
        instance.picture.save("image.jpg", file, save=False)
        # instance.save()
        instance.save()
        confirmation = True
    except Exception as e:
        print(232332, "Saving Failed Error: ", e)

    data = {"saved": confirmation}
    print(data)
    return data
