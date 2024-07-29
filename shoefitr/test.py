import cv2
import numpy as np
import base64

scale = 3
wP = 210 * scale
hP = 297 * scale
us_wP = 216 * scale
us_hP = 280 * scale
a4_ratio = hP/wP
us_ratio = us_hP/us_wP

show_canny = False

def getContours(img, cThr=[75, 75]):

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, cThr[0], cThr[1])
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=1)
    imgThre = cv2.erode(imgDial, kernel, iterations=1)
    # img = cv2.resize(imgThre, (0, 0), None, 0.2, 0.2)

    contours, hiearchy = cv2.findContours(
        imgThre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    biggest, maxArea, paper = biggestContour(contours)  # FIND THE BIGGEST CONTOUR
    print('Area:', maxArea)
    if paper is not None:
        contour = reorder(biggest)
        return contour, paper
    elif show_canny:
        return imgThre, 1
    else:
        return paper, paper

def find_ratio(approx):
    contour = reorder(approx)
    a = contour[0][0]
    b = contour[1][0]
    c = contour[2][0]
    d = contour[3][0]
    length = np.linalg.norm(c-a)
    width = np.linalg.norm(b-a)
    length2 = np.linalg.norm(d-b)
    print('Ratio1: ',length/width)
    diff = length2 - length
    if -0.15*length < diff < 0.15*length:
        ratio = length/width
    else:
        ratio = 0
    print('Length: ', length,'Length2: ', length2,'Width: ', width)
    print('Ratio2: ',ratio)
    print()
    return ratio

def to_base64(im):
    _, im_arr = cv2.imencode('.jpg', im)

    im_bytes = im_arr.tobytes()

    encoded_string = base64.b64encode(im_bytes)
    base64_string = encoded_string.decode('UTF-8')
    base64_string = 'data:image/jpeg;base64,'+base64_string
    return base64_string


def biggestContour(contours):
    biggest = None
    max_area = 0
    paper = None
    ratio = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 3000:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) >= 4:
                ratio = find_ratio(approx)

                if ratio > 1.2 and ratio < 1.60:
                    biggest = approx
                    max_area = area
                    paper = 'a4'
                elif ratio > 1.3 and ratio <=1.30:
                    biggest = approx
                    max_area = area
                    paper = 'us'
    print('ratio: ',ratio)
    print('paper: ',paper)
    print('biggest: ',biggest)
    return biggest, max_area, paper


def reorder(myPoints):
    # print(myPoints.shape)
    left = sorted(myPoints, reverse=False, key=lambda x: x[0][0])[:2]
    right = sorted(myPoints, reverse=True, key=lambda x: x[0][0])[:2]
    print('Left: ',left,'right: ',right)
    if left[0][0][1] > left[1][0][1]:
        c = left[0]
        a = left[1]
    else:
        c = left[1]
        a = left[0]
    if right[0][0][1] > right[1][0][1]:
        d = right[0]
        b = right[1]
    else:
        d = right[1]
        b = right[0]
    pointsNew = np.array([a, b, c, d])
    return pointsNew


def warpImg(img, points, w, h, pad=20):
    # print(points)
    points = reorder(points)
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarp = cv2.warpPerspective(img, matrix, (w, h))
    imgWarp = imgWarp[pad:imgWarp.shape[0]-pad, pad:imgWarp.shape[1]-pad]
    return imgWarp


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged


def findLength(im, paper):
    lower_white = np.array([100, 100, 100], dtype=np.uint8)

    upper_white = np.array([255, 255, 255], dtype=np.uint8)

    # Threshold the HSV image to get only white colors
    mask = cv2.inRange(im, lower_white, upper_white)
    # Bitwise-AND mask and original image

    img = cv2.bitwise_and(im, im, mask=mask)

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 0)
    imgCanny = cv2.Canny(imgBlur, 60, 60)
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=1)
    imgThre = cv2.erode(imgDial, kernel, iterations=1)
    #imgCanny = auto_canny(imgBlur)
    #cv2.imshow('Thre', imgThre)
    imgCanny = imgThre
    found = False

    ll = len(imgCanny)
    ww = len(imgCanny[0])
    for i, row in enumerate(imgCanny):
        for j, col in enumerate(row):
            if i > 0.05*ll and i<0.7*ll and j > 0.1*ww and j < 0.9*ww:
                if col:
                    found = True
                    #print(i, j)
                    break
        if found:
            break
    ratio = hP/len(im)
    #print('ratio', ratio)
    length = len(im) - i
    length = round(length*ratio/scale)
    # cv2.drawMarker(im, (j, i), (0, 0, 255), markerType=cv2.MARKER_STAR,
    #              markerSize=40, thickness=2, line_type=cv2.LINE_AA)
    if found:
        cv2.arrowedLine(im, (j, len(im)-1), (j, i),
                        (250, 50, 50), 3, 8, 0, 0.05)
        cv2.putText(im, '{}mm'.format(length), (j + 25, i + 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                    (250, 50, 50), 2)
        if paper == 'a4':
            cv2.putText(im, 'Paper format: A4', (15, int(0.06*ll)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                        (0, 0, 0), 2)
        elif paper == 'us':
            cv2.putText(im, 'Paper format: US Letter', (10, int(0.03*ll)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                        (0, 0, 0), 2)
    else:
        length = 0
    width = 0
    start = 0
    end = 0
    l = len(im[0])
    for i, row in enumerate(imgCanny):
        for j, col in enumerate(row):
            if col and i > 0.2*ll and i<0.75*ll and j > 0.08*ww and j < 0.92*ww:
                for k, c in enumerate(row[::-1]):
                    if c:
                        tempDiff = l-1 - (j+k)
                        if tempDiff > width:
                            width = tempDiff
                            start = (j, i)
                            end = (l-k-1, i)
    width = round(width*ratio/scale)
    if start != 0:
        cv2.arrowedLine(im, start, end,
                        (100, 100, 255), 3, 8, 0, 0.05)
        cv2.arrowedLine(im, end, start,
                        (100, 100, 255), 3, 8, 0, 0.05)
        cv2.putText(im, '{}mm'.format(width), (end[0]-10, end[1] + 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                    (100, 100, 255), 2)
        cv2.putText(im, '{}mm'.format(width), (start[0]-60, start[1] - 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                    (100, 100, 255), 2)

    print('l', length, 'w', width)
    print('End')
    print()
    #print(len(im)//scale, len(im[0])//scale)
    return {'image':im, 'length':length, 'width':width}


def magic(img):

    upper_white = np.array([255, 255, 255], dtype=np.uint8)
    lower = 220
    cont = None
    while cont is None:
        print('---Lower: ',lower)
        lower_white = np.array([lower, lower, lower], dtype=np.uint8)
        mask = cv2.inRange(img, lower_white, upper_white)
        # Bitwise-AND mask and original image
        im = cv2.bitwise_and(img, img, mask=mask)
        cont, paper = getContours(im)
        if lower <= 80:
            break
        lower -= 20


    if paper == 1:
        return cont, True
    if cont is not None:
        print('Lower: ',lower)
        if paper == 'a4':
            imgWarp = warpImg(img, cont, wP, hP)
            return findLength(imgWarp, paper), True
        elif paper == 'us':
            imgWarp = warpImg(img, cont, us_wP, us_hP)
            return findLength(imgWarp, paper), True
        else:
            print(33,'Error')
            return None, False
    else:
        print(44,'Error')
        return None, False
