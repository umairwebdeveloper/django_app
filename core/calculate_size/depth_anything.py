from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
import numpy as np
from PIL import Image
import requests
import numpy as np
from PIL import Image
# import open3d as o3d
from pathlib import Path
import os
import gradio as gr
import cv2
import torch
import matplotlib.pyplot as plt
from .DepthAnythingV2.depth_anything_v2.dpt import DepthAnythingV2


def predict_depthany_v2(image,x,y):
    model = DepthAnythingV2(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024])
    model.load_state_dict(torch.load('checkpoints/depth_anything_v2_vitl.pth', map_location='cpu'))
    model.eval()
    depth = model.infer_image(image) # HxW raw depth map
    depth_value = depth[y, x]
    depth = cv2.circle(depth, (x,y), radius=7, color=(25,25, 225), thickness=-1)
    # plt.imshow(depth)
    # plt.pause(0.00001)
    # plt.show()
    return depth_value

    

# img, gltf_path, gltf_path = process_image(image_path=image_path)


