import open3d as o3d
import numpy as np
import cv2
import gradio as gr
from transformers import DPTFeatureExtractor, DPTForDepthEstimation
import torch
import numpy as np
from PIL import Image
import open3d as o3d
from pathlib import Path
import os
from .foot_shape_estimation import FootShapeExtractor
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def invert_matrix(matrix):
    """
    Inverts a given square matrix using NumPy.

    Parameters:
    matrix (np.ndarray): A square matrix to be inverted.

    Returns:
    np.ndarray: The inverted matrix.

    Raises:
    ValueError: If the matrix is singular and cannot be inverted.
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("Input must be a NumPy array.")
    
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input must be a square matrix (2D array with equal dimensions).")
    
    try:
        inverse_matrix = np.linalg.inv(matrix)
        return inverse_matrix
    except np.linalg.LinAlgError:
        raise ValueError("The matrix is singular and cannot be inverted.")
shape_extractor = FootShapeExtractor()
feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-large")
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")

def depth_to_point_cloud(depth_map, intrinsic_matrix, depth_scale=1000.0, depth_trunc=1000.0):
    """
    Convert a depth map to a 3D point cloud.
    
    Parameters:
    - depth_map: 2D numpy array of depth values.
    - intrinsic_matrix: 3x3 camera intrinsic matrix.
    - depth_scale: Scaling factor for depth values (default is 1000.0 for converting millimeters to meters).
    - depth_trunc: Truncate depth values beyond this value (default is 1000.0).
    
    Returns:
    - point_cloud: Open3D point cloud object.
    """
    height, width = depth_map.shape
    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]

    # Create mesh grid for image coordinates
    x = np.linspace(0, width - 1, width)
    y = np.linspace(0, height - 1, height)
    u, v = np.meshgrid(x, y)

    # Convert depth map to 3D points
    z = depth_map / depth_scale
    z[z > depth_trunc] = 0  # Truncate depth values
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)

    # Remove points with zero depth
    points = points[z.flatten() > 0]

    # Create Open3D point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    # o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    return point_cloud


def process_image(image_path):
    image_path = Path(image_path)
    image_raw = Image.open(image_path)
    image = image_raw.resize(
        (800, int(800 * image_raw.size[1] / image_raw.size[0])),
        Image.Resampling.LANCZOS)

    # prepare image for the model
    encoding = feature_extractor(image, return_tensors="pt")
    mask_image = shape_extractor.predict_shape_mask(image)
    cv2.imshow("f",mask_image)
    # forward pass
    with torch.no_grad():
        outputs = model(**encoding)
        predicted_depth = outputs.predicted_depth

    # interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    ).squeeze()
    output = prediction.cpu().numpy()
    depth_image = (output * 255 / np.max(output)).astype('uint8')
    print("shape:",depth_image.shape)
    print("shapyue:",type(depth_image))
    mask_image = cv2.resize(mask_image, (depth_image.shape[1], depth_image.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    print("shape:",mask_image.shape)
    # mask_image =cv2.merge([mask_image, mask_image, mask_image])
    masked_image = cv2.bitwise_and(mask_image, depth_image)
    max_value = np.max(masked_image)*3
    new_array = np.full(depth_image.shape, max_value)
    new_depth = new_array - depth_image
    return masked_image


def callculate_cloud(image_raw):
    # image_path = Path(image_path)
    # image_raw = Image.open(image_path)
    # image_raw.show()
    image = image_raw.resize(
        (800, int(800 * image_raw.size[1] / image_raw.size[0])),
        Image.Resampling.LANCZOS)

    # prepare image for the model
    encoding = feature_extractor(image, return_tensors="pt")
    mask_image = shape_extractor.predict_shape_mask(image)
    # cv2.imshow("f",mask_image)
    # forward pass
    with torch.no_grad():
        outputs = model(**encoding)
        predicted_depth = outputs.predicted_depth

    # interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    ).squeeze()
    output = prediction.cpu().numpy()
    depth_image = (output * 255 / np.max(output)).astype('uint8')
    # print("shape:",depth_image.shape)
    # print("shapyue:",type(depth_image))
    mask_image = cv2.resize(mask_image, (depth_image.shape[1], depth_image.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    # print("shape:",mask_image.shape)
    # mask_image =cv2.merge([mask_image, mask_image, mask_image])
    masked_image = cv2.bitwise_and(mask_image, depth_image)
    max_value = np.max(masked_image)*3
    new_array = np.full(depth_image.shape, max_value)
    new_depth = new_array - depth_image
    # custom_mesh(depth_image)
    # print("shape:",mask_image.shape)
    # print("shape:",new_depth.shape)
    # new_depth = cv2.bitwise_and(mask_image, new_depth)
    # Step 3: Define a threshold
    # threshold = 10  # You can set this to any value you want

    # # Step 4: Add the threshold to the maximum pixel value
    # max_plus_threshold = max_value -10

    # # Step 5: Subtract this value from each pixel in the image
    # # modified_image_array = depth_image - max_plus_threshold

    # print(type(depth_image))
    return masked_image


def custom_mesh(depth_image):
    height, width = depth_image.shape
    x = np.arange(0, width)
    y = np.arange(0, height)
    x, y = np.meshgrid(x, y)
    z = depth_image

    # Normalize z for better visualization
    z = cv2.normalize(z, None, 0, 255, cv2.NORM_MINMAX)

    # Create a 3D mesh grid
    points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T

    # Visualization
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2], cmap='viridis')

    plt.show()

def main():
    # Load depth map (assuming a single-channel image where depth is in millimeters)
    image_name = 'frame_1094.jpg'
    depth_map = process_image(image_name)

    # Define camera intrinsic matrix (example values, replace with your actual values)
    intrinsic_matrix = np.array([[525.0, 0.0, 319.5],
                                 [0.0, 525.0, 239.5],
                                 [0.0, 0.0, 1.0]])

    # Convert depth map to point cloud
    point_cloud = depth_to_point_cloud(depth_map, intrinsic_matrix)
    # Save and visualize the point cloud
    new_image_name = image_name.replace('.jpg', '.ply')
    o3d.io.write_point_cloud(new_image_name, point_cloud)
    o3d.visualization.draw_geometries([point_cloud])

    # point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # # Poisson reconstruction
    # mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=8)

    # # Crop the mesh
    # bbox = point_cloud.get_axis_aligned_bounding_box()
    # mesh = mesh.crop(bbox)

    # # Save the mesh
    # o3d.io.write_triangle_mesh("output_mesh.ply", mesh)

    # # Visualize the mesh
    # o3d.visualization.draw_geometries([mesh])

if __name__ == "__main__":
    main()



def process_image2(image_path):
    image_path = Path(image_path)
    image_raw = Image.open(image_path)
    image = image_raw.resize(
        (800, int(800 * image_raw.size[1] / image_raw.size[0])),
        Image.Resampling.LANCZOS)

    # prepare image for the model
    encoding = feature_extractor(image, return_tensors="pt")
    mask_image = shape_extractor.predict_shape_mask(image)
    # cv2.imshow("f",mask_image)
    # forward pass
    with torch.no_grad():
        outputs = model(**encoding)
        predicted_depth = outputs.predicted_depth

    # interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    ).squeeze()
    output = prediction.cpu().numpy()
    depth_image = (output * 255 / np.max(output)).astype('uint8')
    # print("shape:",depth_image.shape)
    # print("shapyue:",type(depth_image))
    mask_image = cv2.resize(mask_image, (depth_image.shape[1], depth_image.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    # print("shape:",mask_image.shape)
    # mask_image =cv2.merge([mask_image, mask_image, mask_image])
    masked_image = cv2.bitwise_and(mask_image, depth_image)
    max_value = np.max(masked_image)*1
    new_array = np.full(depth_image.shape, max_value)
    
    # print("shape new_array:",new_array.shape)
    # print("shape mask_image:",mask_image.shape)

    # print("shape new_array:",new_array.dtype)
    # print("shape mask_image:",mask_image.dtype)
    zero_array = np.zeros_like(depth_image)
    new_array = new_array.astype(depth_image.dtype)
    mask_image = mask_image.astype(depth_image.dtype)
    absolute_image = cv2.absdiff(zero_array, depth_image)

    depth_image[mask_image == 255] = absolute_image[mask_image == 255]

    # new_array = cv2.bitwise_and(mask_image, new_array)
    # mask_image = cv2.resize(mask_image, (depth_image.shape[1], depth_image.shape[0]), interpolation=cv2.INTER_NEAREST)
    # new_array = cv2.bitwise_and(mask_image, new_array)
    # new_depth = new_array - depth_image
    # new_depth =cv2.subtract(new_array, depth_image, mask=mask_image)
    # print("shape:",mask_image.shape)
    # print("shape:",new_depth.shape)
    # new_depth = cv2.bitwise_and(mask_image, new_depth)
    # Step 3: Define a threshold
    threshold = 10  # You can set this to any value you want

    # Step 4: Add the threshold to the maximum pixel value
    max_plus_threshold = max_value -10

    # Step 5: Subtract this value from each pixel in the image
    # modified_image_array = depth_image - max_plus_threshold

    # print(type(depth_image))
    return depth_image



def calculate_cloud_point_from_pic(image):
    depth_map = callculate_cloud(image)

    # Define camera intrinsic matrix (example values, replace with your actual values)
    intrinsic_matrix = np.array([[525.0, 0.0, 319.5],
                                 [0.0, 525.0, 239.5],
                                 [0.0, 0.0, 1.0]])

    # Convert depth map to point cloud
    point_cloud = depth_to_point_cloud(depth_map, intrinsic_matrix)
    return point_cloud