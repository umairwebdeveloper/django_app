import cv2
import torch
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import shutil
from scipy.interpolate import RectBivariateSpline

#To Clear the model cache
# shutil.rmtree(torch.hub.get_dir(), ignore_errors=True)


# # #download the model for small model
# midas = torch.hub.load('intel-isl/MiDaS','MiDaS_small')
# midas.to('cpu')
# midas.eval()

# #Process image
# transforms = torch.hub.load('intel-isl/MiDaS','transforms')
# transform = transforms.small_transform

#for large model

def convert(box):
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    return int(x),int(y)

def complex_equation_mapping(x):
    """
    A complex equation mapping x to y.
    For example, using a quadratic equation.
    """
    y = 2*x**2 + 3*x + 5
    return y

midas = torch.hub.load('intel-isl/MiDaS','DPT_Large')
midas.to('cpu')
midas.eval()

#Process image
transforms = torch.hub.load('intel-isl/MiDaS','transforms')
transform = transforms.dpt_transform



alpha = 0.2
previous_depth = 0.0
depth_scale = 1.0

#Applying exponential moving average filter
def apply_ema_filter(current_depth):
    global previous_depth
    filtered_depth = alpha * current_depth + (1 - alpha) * previous_depth
    previous_depth = filtered_depth  # Update the previous depth value
    return filtered_depth


#Define depth to distance
def depth_to_distance(depth_value,depth_scale):
    return 1.0 / (depth_value*depth_scale)

def depth_to_distance1(depth_value,depth_scale):
    return -1.0 / (depth_value*depth_scale)

def predict(img,midas,mid_y,mid_x):
    imgbatch = transform(img).to('cpu')

    # Making a prediction
    with torch.no_grad():
        prediction = midas(imgbatch)
        ppp = prediction
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode='bicubic',
            align_corners=False
        ).squeeze()

    output = prediction.cpu().numpy()
    output_norm = cv2.normalize(output, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Creating a spline array of non-integer grid
    h, w = output_norm.shape
    x_grid = np.arange(w)
    y_grid = np.arange(h)

    # Create a spline object using the output_norm array
    spline = RectBivariateSpline(y_grid, x_grid, output_norm)
    depth_mid_filt = spline(mid_y,mid_x)
    depth_midas = depth_to_distance(depth_mid_filt, depth_scale)
    depth_mid_filt = (apply_ema_filter(depth_midas)/10)[0][0]
    img = cv2.circle(img, (mid_x,mid_y), radius=7, color=(255,25, 25), thickness=-1)
    cv2.putText(img, "Distance in unit: " + str(
        np.format_float_positional(depth_mid_filt , precision=3)) , (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 3)
    return img,depth_mid_filt
    