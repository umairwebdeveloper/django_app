import open3d as o3d
import numpy as np
import alphashape
from collections import defaultdict
from scipy.spatial import Delaunay, ConvexHull, KDTree, distance
from scipy.interpolate import splprep, splev
from matplotlib.path import Path
from itertools import combinations
import subprocess
import os




def read_parameters_from_file(file_path):
    parameters = {}
    with open(file_path, 'r') as file:
        for line in file:
            key, value = line.split('=')
            key = key.strip()
            value = value.strip()
            if key in ["smooth_iterations"]:
                parameters[key] = float(value)
            else:
                parameters[key] = value
    return parameters

# Construct file path dynamically
script_dir = os.path.dirname(__file__)  # Directory of the script
file_path = os.path.join(script_dir, 'parameters.txt')
# Read parameters from text file
params = read_parameters_from_file(file_path)



#-----------------------------------------------------------------------------------------------------------------------------------------
# Load the point cloud
ply_path = params['ply_file_path']
ply_path_dir = os.path.dirname(ply_path)

# Define paths
cloudcompare_path = os.path.join(script_dir, 'CloudCompare_v2.13.2.preview_bin_x64\\CloudCompare.exe')  # Adjust path to CloudCompare executable
temp_file = os.path.join(ply_path_dir,"temp.ply")
#output_ply_sub_file = os.path.join(ply_path_dir,"output_ply_sub.ply")

output_ply_sub_file = os.path.join(ply_path_dir, os.path.splitext(os.path.basename(ply_path))[0] + '_processed.ply')

# Define a single command for all operations
cloudcompare_command = [
    cloudcompare_path,
    '-SILENT',
    '-O', ply_path,
    '-AUTO_SAVE', 'OFF',
    '-NO_TIMESTAMP',
    '-SS', 'OCTREE', '6',
    '-SOR', '20', '1.00',
    '-SOR', '20', '1.00',
    '-C_EXPORT_FMT', 'PLY',
    '-SAVE_CLOUDS', 'FILE', output_ply_sub_file,
]

# Function to run the command
def run_cloudcompare_command(command):
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print("CloudCompare Output:")
    print(result.stdout)
    print("CloudCompare Errors:")
    print(result.stderr)

# Run the CloudCompare command
run_cloudcompare_command(cloudcompare_command)

# Check if the output file was created
if os.path.exists(output_ply_sub_file):
    print(f"Processed point cloud saved to {output_ply_sub_file}")
else:
    print("Processing failed or the file was not saved as expected.")

#print(f"Processed point cloud saved to {output_file}")





#-----------------------------------------------------------------------------------------------------------------------------------------
pcd = o3d.io.read_point_cloud(output_ply_sub_file)

# Visualize the result
#o3d.visualization.draw_geometries([pcd])

# Convert point cloud to numpy array
points = np.asarray(pcd.points)

for _ in range(1):
    # Compute covariance matrix
    cov_matrix = np.cov(points.T)

    # Perform eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Sort eigenvalues and eigenvectors
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Compute center of the point cloud
    center = np.mean(points, axis=0)

    # Transform the point cloud to align with PCA axes
    transformed_points = (points - center) @ eigenvectors

    # Create a bounding box
    min_bounds = np.min(transformed_points, axis=0)
    max_bounds = np.max(transformed_points, axis=0)

    # Construct oriented bounding box (OBB)
    obb_center = center
    obb_extent = (max_bounds - min_bounds) / 2
    obb = o3d.geometry.OrientedBoundingBox(obb_center, eigenvectors, obb_extent)

    # Visualize results
    #o3d.visualization.draw_geometries([pcd, obb])
    points = transformed_points

# Compute the oriented bounding box
#obb = processed_pcd.get_oriented_bounding_box()



Transformed_pcd = o3d.geometry.PointCloud()
Transformed_pcd.points = o3d.utility.Vector3dVector(points)



# Convert point cloud to numpy array
#points = np.asarray(pcd.points)

# Find the bounding box of the point cloud
bbox = Transformed_pcd.get_axis_aligned_bounding_box()
bbox_min = bbox.get_min_bound()
bbox_max = bbox.get_max_bound()

# Divide the point cloud into two halves along its length (x-axis)
mid_x = (bbox_min[0] + bbox_max[0]) / 2
left_half = points[points[:, 0] <= mid_x]
right_half = points[points[:, 0] > mid_x]

# Create point clouds for both halves
left_pcd = o3d.geometry.PointCloud()
left_pcd.points = o3d.utility.Vector3dVector(left_half)
right_pcd = o3d.geometry.PointCloud()
right_pcd.points = o3d.utility.Vector3dVector(right_half)

# Calculate bounding boxes for both halves
left_bbox = left_pcd.get_axis_aligned_bounding_box()
right_bbox = right_pcd.get_axis_aligned_bounding_box()

# Determine which half has a larger width (y-axis)
left_width = left_bbox.get_extent()[1]
right_width = right_bbox.get_extent()[1]
#left_width_x = left_bbox.get_extent()[0]
#right_width_x = right_bbox.get_extent()[0]
#print(f"left_width_x: {left_width_x}")
#print(f"right_width_x: {right_width_x}")

if params['Foot_type'] == 'Left':
    selected_pcd = left_pcd
    selected_bbox = left_bbox
else:
    selected_pcd = right_pcd
    selected_bbox = right_bbox

# Visualize the result
#o3d.visualization.draw_geometries([selected_pcd])

# Find the bottommost point of the selected half in the reverse direction (maximum bound)
bottom_point = selected_bbox.get_max_bound()

# Create a plane at the bottommost point
plane_center = bottom_point
plane_normal = np.array([0, 0, 1])

# Project points onto the plane
selected_points = np.asarray(selected_pcd.points)
plane_normal = plane_normal / np.linalg.norm(plane_normal)  # Normalize the plane normal

# Remove original points that were selected
back_part_points = np.array([pt for pt in points if pt.tolist() not in selected_points.tolist()])

# Find the point with the minimum distance to the plane
distances_to_plane = np.dot(selected_points - plane_center, plane_normal)
min_distance_index = np.argmin(np.abs(distances_to_plane))
min_distance_point = selected_points[min_distance_index]

# Divide the selected point cloud into two parts from the minimum distance point
dividing_line_x = min_distance_point[0]
if params['Foot_type'] == 'Left':
    part_to_project = selected_points[selected_points[:, 0] <= dividing_line_x]
    remaining_part = selected_points[selected_points[:, 0] > dividing_line_x]
else:
    part_to_project = selected_points[selected_points[:, 0] >= dividing_line_x]
    remaining_part = selected_points[selected_points[:, 0] < dividing_line_x]

# Project the points from the part to be projected
projected_points = part_to_project.copy()

for i, point in enumerate(part_to_project):
    vector_from_point_to_plane = point - plane_center
    distance_to_plane = np.dot(vector_from_point_to_plane, plane_normal)
    projected_points[i] = point - distance_to_plane * plane_normal

projected_pcd = o3d.geometry.PointCloud()
projected_pcd.points = o3d.utility.Vector3dVector(projected_points)

# Visualize the result
#o3d.visualization.draw_geometries([projected_pcd])

# Calculate the adjustment direction and factor
back_points_min_x = np.min(back_part_points[:, 0])
back_points_max_x = np.max(back_part_points[:, 0])
projected_points_min_x = np.min(projected_points[:, 0])
projected_points_max_x = np.max(projected_points[:, 0])

if params['Foot_type'] == 'Left':
    overall_min_x = projected_points_max_x
    overall_max_x = back_points_min_x
else:
    overall_min_x = projected_points_min_x
    overall_max_x = back_points_max_x

# Adjust remaining points to be closer to the plane near the boundary
adjusted_remaining_part = remaining_part.copy()

for i, point in enumerate(remaining_part):
    if i == 0 or i == 1:
        continue
    
    distance_to_plane = np.dot(point - plane_center, plane_normal)
    factor = (point[0] - overall_min_x) / (overall_max_x - overall_min_x)  # Factor to adjust the distance based on x-coordinate
    adjusted_distance_to_plane = distance_to_plane * (1 - factor * 0.8)  # Adjust points closer to the plane
    adjusted_remaining_part[i] = point - adjusted_distance_to_plane * plane_normal

adjusted_remaining_pcd = o3d.geometry.PointCloud()
adjusted_remaining_pcd.points = o3d.utility.Vector3dVector(adjusted_remaining_part)

# Step 1: Process Projected Points

# Get the 2D points by dropping the z-coordinate from the projected points
projected_2d_points = projected_points[:, :2]

# Compute the convex hull of the 2D points
hull = ConvexHull(projected_2d_points)
hull_points = projected_2d_points[hull.vertices]

# Ensure boundary curve points lie on the plane
boundary_curve_points = np.column_stack((hull_points[:, 0], hull_points[:, 1], np.zeros_like(hull_points[:, 0]) + plane_center[2]))

# Use spline interpolation to smooth the boundary curve
hull_points_2d = hull_points.copy()
hull_points_2d = np.vstack([hull_points_2d, hull_points_2d[0]])  # Close the loop
tck, u = splprep([hull_points_2d[:, 0], hull_points_2d[:, 1]], s=0)
unew = np.linspace(0, 1, len(hull_points) * 3)  # 3 times the original number of points
out = splev(unew, tck)
smoothed_boundary_curve_2d = np.array(out).T

# Add z-coordinate to smoothed boundary curve points
smoothed_boundary_curve_points = np.column_stack((smoothed_boundary_curve_2d[:, 0], smoothed_boundary_curve_2d[:, 1], np.zeros_like(smoothed_boundary_curve_2d[:, 0]) + plane_center[2]))

# Step 2: Refine Projected Points Inside the Boundary

# Create a grid of points within the boundary
grid_spacing = 0.005  # Adjust the spacing as needed
x_min, x_max = np.min(smoothed_boundary_curve_2d[:, 0]), np.max(smoothed_boundary_curve_2d[:, 0])
y_min, y_max = np.min(smoothed_boundary_curve_2d[:, 1]), np.max(smoothed_boundary_curve_2d[:, 1])
x_grid, y_grid = np.meshgrid(np.arange(x_min, x_max, grid_spacing), np.arange(y_min, y_max, grid_spacing))
grid_points = np.vstack([x_grid.ravel(), y_grid.ravel()]).T

# Filter grid points to keep only those inside the boundary
hull_path = Path(smoothed_boundary_curve_2d)
inside_mask = hull_path.contains_points(grid_points)
inside_points = grid_points[inside_mask]

# Add z-coordinate to inside points
inside_points_3d = np.column_stack((inside_points[:, 0], inside_points[:, 1], np.zeros_like(inside_points[:, 0]) + plane_center[2]))

# Create point cloud for inside points
inside_pcd = o3d.geometry.PointCloud()
inside_pcd.points = o3d.utility.Vector3dVector(inside_points_3d)

# Combine the new boundary set with adjusted_remaining_part and back_part_points
combined_points_step1 = np.vstack((smoothed_boundary_curve_points, inside_points_3d, adjusted_remaining_part, back_part_points))

# Create point cloud from combined points
combined_pcd_step1 = o3d.geometry.PointCloud()
combined_pcd_step1.points = o3d.utility.Vector3dVector(combined_points_step1)

# Visualize the result
#o3d.visualization.draw_geometries([combined_pcd_step1])



#-----------------------------------------------------------------------------------------------------------------------------------------
# Step 1: Calculate the center of the back part points
center_of_back_part = np.mean(back_part_points, axis=0)

# Step 2: Create the rotation matrix for 180 degrees around the x-axis
rotation_matrix = np.array([
    [1, 0, 0],
    [0, -1, 0],
    [0, 0, -1]
])

# Step 3: Apply the rotation matrix to the back part points
centered_back_part_points = back_part_points - center_of_back_part  # Translate to origin
rotated_back_part_points = centered_back_part_points @ rotation_matrix.T  # Apply rotation
rotated_back_part_points += center_of_back_part  # Translate back


# Sort points by z-coordinate
sorted_indices = np.argsort(rotated_back_part_points[:, 2])
sorted_points = rotated_back_part_points[sorted_indices]

# Find the maximum extent in the x or y direction for each z-level
max_extent_indices = []
max_extent = -np.inf

for i in range(len(sorted_points)):
    point = sorted_points[i]
    current_extent = np.linalg.norm(point[:2])  # Calculate the extent in the x-y plane

    if current_extent > max_extent:
        max_extent = current_extent
        max_extent_indices.append(i)

# Remove points beyond the maximum curvature point
curvature_change_index = max_extent_indices[-1]
filtered_points = sorted_points[:curvature_change_index + 1]

# Sort points by x-coordinate and divide into two halves
mid_x = np.mean(filtered_points[:, 0])
first_half = filtered_points[filtered_points[:, 0] <= mid_x]
second_half = filtered_points[filtered_points[:, 0] > mid_x]

# For the first half:
# Divide into 2 segments along the y-axis
mid_y_first_half = np.mean(first_half[:, 1])
first_half_y1 = first_half[first_half[:, 1] <= mid_y_first_half]
first_half_y2 = first_half[first_half[:, 1] > mid_y_first_half]

def process_segments(segment):
    # Divide into 10 segments along the x-axis
    x_min, x_max = np.min(segment[:, 0]), np.max(segment[:, 0])
    x_segments = np.linspace(x_min, x_max, 11)
    segments = [segment[(segment[:, 0] >= x_segments[i]) & (segment[:, 0] < x_segments[i + 1])] for i in range(10)]
    return segments

first_half_segments_y1 = process_segments(first_half_y1)
first_half_segments_y2 = process_segments(first_half_y2)

# Further divide the first segment of these 10 segments into 3 segments along the y-axis
def subdivide(segment):
    y_min, y_max = np.min(segment[:, 1]), np.max(segment[:, 1])
    y_segments = np.linspace(y_min, y_max, 4)
    subsegments = [segment[(segment[:, 1] >= y_segments[i]) & (segment[:, 1] < y_segments[i + 1])] for i in range(3)]
    return subsegments

subsegments_y1 = subdivide(first_half_segments_y1[0])
subsegments_y2 = subdivide(first_half_segments_y2[0])

all_segments = first_half_segments_y1 + first_half_segments_y2 + subsegments_y1 + subsegments_y2

# Calculate the mean of the maximum height of these 24 segments
max_heights = [np.max(seg[:, 2]) for seg in all_segments if len(seg) > 0]
mean_max_height = np.mean(max_heights)

boundary_points = []
Remove_points = []

# Create a new point at this mean height for each segment and find points above this height
for segment in all_segments:
    if len(segment) == 0:
        continue

    below_mask = segment[:, 2] < mean_max_height
    below_points = segment[below_mask]
    if len(below_points) < 2:
        continue

    first_point = below_points[np.argmax(below_points[:, 2])]
    below_points = below_points[below_points[:, 2] < first_point[2]]
    if len(below_points) == 0:
        continue

    second_point = below_points[np.argmax(below_points[:, 2])]
    x_diff = first_point[0] - second_point[0]
    y_diff = first_point[1] - second_point[1]

    new_point = np.array([[first_point[0] + x_diff, first_point[1] + y_diff, mean_max_height]])
    boundary_points.append(new_point)

    above_mask = segment[:, 2] > mean_max_height
    above_points = segment[above_mask]
    Remove_points.append(above_points)

# For the second half:
# Divide into 2 segments along the y-axis
mid_y_second_half = np.mean(second_half[:, 1])
second_half_y1 = second_half[second_half[:, 1] <= mid_y_second_half]
second_half_y2 = second_half[second_half[:, 1] > mid_y_second_half]

second_half_segments_y1 = process_segments(second_half_y1)
second_half_segments_y2 = process_segments(second_half_y2)

# Find the maximum height point for each segment
for segment in second_half_segments_y1 + second_half_segments_y2:
    if len(segment) == 0:
        continue
    max_point = segment[np.argmax(segment[:, 2])]
    boundary_points.append(max_point.reshape(1, -1))

# Concatenate all boundary points
#boundary_points = np.vstack(boundary_points)

#-----------------------------------------------------------------------------------------------------------------------------------------
#rotate other point clouds by 180 deg for adjusted_remaining_part
#center_of_adjusted_remaining_part= np.mean(adjusted_remaining_part, axis=0)
centered_adjusted_remaining_part_points = adjusted_remaining_part - center_of_back_part  # Translate to origin
rotated_adjusted_remaining_part_points = centered_adjusted_remaining_part_points @ rotation_matrix.T  # Apply rotation
rotated_adjusted_remaining_part_points += center_of_back_part  # Translate back

# Divide above into 2 segments along the y-axis
mid_y_adjusted_remaining_part = np.mean(rotated_adjusted_remaining_part_points[:, 1])
adjusted_remaining_part_y1 = rotated_adjusted_remaining_part_points[rotated_adjusted_remaining_part_points[:, 1] <= mid_y_adjusted_remaining_part]
adjusted_remaining_part_y2 = rotated_adjusted_remaining_part_points[rotated_adjusted_remaining_part_points[:, 1] > mid_y_adjusted_remaining_part]

adjusted_remaining_part_segments_y1 = process_segments(adjusted_remaining_part_y1)
adjusted_remaining_part_segments_y2 = process_segments(adjusted_remaining_part_y2)

# Find the maximum height point for each segment
for segment in adjusted_remaining_part_segments_y1 + adjusted_remaining_part_segments_y2:
    if len(segment) == 0:
        continue
    max_point = segment[np.argmax(segment[:, 2])]
    boundary_points.append(max_point.reshape(1, -1))

# Concatenate all boundary points
boundary_points = np.vstack(boundary_points)
# Concatenate all points to remove
Remove_points = np.vstack(Remove_points)
filtered_points = np.array([point for point in filtered_points if not any(np.all(point == remove_point) for remove_point in Remove_points)])

#-----------------------------------------------------------------------------------------------------------------------------------------
#rotate other point clouds by 180 deg for inside_points_3d
#center_of_inside_points_3d= np.mean(inside_points_3d, axis=0)
centered_inside_points_3d_points = inside_points_3d - center_of_back_part  # Translate to origin
rotated_inside_points_3d_points = centered_inside_points_3d_points @ rotation_matrix.T  # Apply rotation
rotated_inside_points_3d_points += center_of_back_part  # Translate back

#rotate other point clouds by 180 deg for smoothed_boundary_curve_points
#center_of_smoothed_boundary_curve_points= np.mean(smoothed_boundary_curve_points, axis=0)
centered_smoothed_boundary_curve_points = smoothed_boundary_curve_points - center_of_back_part  # Translate to origin
rotated_smoothed_boundary_curve_points = centered_smoothed_boundary_curve_points @ rotation_matrix.T  # Apply rotation
rotated_smoothed_boundary_curve_points += center_of_back_part  # Translate back


#-----------------------------------------------------------------------------------------------------------------------------------------
def filter_points_by_x(points, tolerance=0.0005):
    # Sort points by x-values
    points = points[np.argsort(points[:, 0])]

    # Group points by x-values within a tolerance
    grouped_points = defaultdict(list)
    current_group = []
    current_x = points[0][0]

    for point in points:
        if abs(point[0] - current_x) <= tolerance:
            current_group.append(point)
        else:
            # Assign the current group to the grouped_points dictionary
            if current_group:
                mean_x = np.mean([p[0] for p in current_group])
                grouped_points[mean_x].extend(current_group)
            # Start a new group
            current_group = [point]
            current_x = point[0]

    # Add the last group if not empty
    if current_group:
        mean_x = np.mean([p[0] for p in current_group])
        grouped_points[mean_x].extend(current_group)

    # Filter points to keep only those with the largest and smallest y-values for each x-value group
    filtered_points = []
    for x_val, group in grouped_points.items():
        group = np.array(group)
        max_y_point = group[np.argmax(group[:, 1])]
        min_y_point = group[np.argmin(group[:, 1])]
        filtered_points.extend([max_y_point, min_y_point])

    return np.array(filtered_points)

# Apply the filtering
filtered_boundary_points = filter_points_by_x(rotated_smoothed_boundary_curve_points)

# Concatenate with boundary_points
combined_points_step2 = np.vstack((filtered_boundary_points, boundary_points))
#combined_points_step2 = np.vstack((boundary_points))

# Function to apply Laplacian smoothing
def laplacian_smoothing(points, iterations=10, lambda_val=0.9):
    smoothed_points = points.copy()
    for _ in range(iterations):
        new_points = smoothed_points.copy()
        for i in range(len(points)):
            # Find neighbors within a certain distance threshold
            neighbors = np.where(np.linalg.norm(smoothed_points - smoothed_points[i], axis=1) < 0.001)[0]
            if len(neighbors) > 0:
                new_points[i] = smoothed_points[i] + lambda_val * (np.mean(smoothed_points[neighbors], axis=0) - smoothed_points[i])
        smoothed_points = new_points
    return smoothed_points

# Apply Laplacian smoothing to the combined points
smoothed_combined_points = laplacian_smoothing(combined_points_step2)

# Create point clouds for visualization
smoothed_pcd = o3d.geometry.PointCloud()
smoothed_pcd.points = o3d.utility.Vector3dVector(smoothed_combined_points)

# Visualize the filtered and smoothed boundaries
#o3d.visualization.draw_geometries([smoothed_pcd])

#-----------------------------------------------------------------------------------------------------------------------------------------

# Step 2: Create a new plane 0.02 units below the previous plane
#distance_to_plane = np.dot(center_of_back_part - plane_center, plane_normal)
new_plane_center = rotated_inside_points_3d_points[0] - np.array([0, 0, 0.004])
new_plane_normal = plane_normal

def interpolate_to_plane_z(smoothed_points, plane_z, num_intervals=10):
    
    all_interpolated_points = []

    for sp in (smoothed_points):
        # Calculate the difference in z-coordinates
        z_diff = plane_z - sp[2]
        
        # Interpolate points between the original and the plane's z-value
        for i in range(num_intervals + 1):
            t = i / num_intervals
            interpolated_point = np.array([sp[0], sp[1], sp[2] + t * z_diff])
            all_interpolated_points.append(interpolated_point)

    return np.array(all_interpolated_points)

# Interpolate points along the z-axis to the plane's z-value
plane_z = new_plane_center[2]
interpolated_points = interpolate_to_plane_z(smoothed_combined_points, plane_z)
interpolated_pcd = o3d.geometry.PointCloud()
interpolated_pcd.points = o3d.utility.Vector3dVector(interpolated_points)

filtered_points_pcd = o3d.geometry.PointCloud()
filtered_points_pcd.points = o3d.utility.Vector3dVector(filtered_points)

# Visualize the results
#o3d.visualization.draw_geometries([smoothed_pcd, projected_boundary_pcd, interpolated_pcd, inside_points_pcd, filtered_points_pcd])

#Combine all different points.
combined_points_step3 = np.vstack((rotated_inside_points_3d_points, rotated_adjusted_remaining_part_points, filtered_points, smoothed_combined_points))
#combined_points_step3 = np.vstack((rotated_adjusted_remaining_part_points, filtered_points, smoothed_combined_points))

combined_points_step3_pcd = o3d.geometry.PointCloud()
combined_points_step3_pcd.points = o3d.utility.Vector3dVector(combined_points_step3)


#-----------------------------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------------------------
'''
def create_grid_within_boundary(boundary_points, spacing=0.005):
    # Determine the bounding box of the boundary points
    min_bounds = np.min(boundary_points, axis=0)
    max_bounds = np.max(boundary_points, axis=0)
    
    # Create a grid of points within the bounding box
    x_range = np.arange(min_bounds[0], max_bounds[0], spacing)
    y_range = np.arange(min_bounds[1], max_bounds[1], spacing)
    z_val = np.mean(boundary_points[:, 2])  # Use a constant z-value for the grid points
    
    grid_points = np.array([[x, y, z_val] for x in x_range for y in y_range])
    
    # Create a Path object for the boundary
    boundary_path = Path(boundary_points[:, :2])
    
    # Check which grid points are inside the boundary
    mask = boundary_path.contains_points(grid_points[:, :2])
    
    # Filter the grid points to keep only those inside the boundary
    inside_points = grid_points[mask]
    
    return inside_points


# Create points inside the projected boundary points
inside_points = create_grid_within_boundary(projected_boundary_points)

inside_points_pcd = o3d.geometry.PointCloud()
inside_points_pcd.points = o3d.utility.Vector3dVector(inside_points)
'''


#-----------------------------------------------------------------------------------------------------------------------------------------
# Visualize the results
#o3d.visualization.draw_geometries([combined_points_step3_pcd])

#Combine for side wall surface points
#Side_wall_points = np.vstack((smoothed_combined_points, projected_boundary_points, interpolated_points))
#Side_wall_points_pcd = o3d.geometry.PointCloud()
#Side_wall_points_pcd.points = o3d.utility.Vector3dVector(Side_wall_points)


#-----------------------------------------------------------------------------------------------------------------------------------------
# Create Delaunay triangulation
tri = Delaunay(combined_points_step3[:, :2])
main_surf_triangles = tri.simplices

#tri_1 = Delaunay(Side_wall_points[:, :2])
#side_boundary_triangles = tri_1.simplices


# Define the maximum edge length
max_edge_length = 0.02  # Adjust this value as needed

# Filter large faces
def filter_large_faces(triangles, points, max_edge_length):
    filtered_triangles = []
    for tri in triangles:
        p0, p1, p2 = points[tri]
        edges = [np.linalg.norm(p1 - p0), np.linalg.norm(p2 - p1), np.linalg.norm(p0 - p2)]
        if all(edge <= max_edge_length for edge in edges):
            filtered_triangles.append(tri)
    return np.array(filtered_triangles)

# Filter the triangles
filtered_triangles_main = filter_large_faces(main_surf_triangles, np.asarray(combined_points_step3_pcd.points), max_edge_length)
#filtered_triangles_side_wall = filter_large_faces(side_boundary_triangles, np.asarray(Side_wall_points_pcd.points), max_edge_length)


# Create a mesh from the filtered triangles for main surface
vertices_main_surf = o3d.utility.Vector3dVector(np.asarray(combined_points_step3_pcd.points))
filtered_triangles_main = o3d.utility.Vector3iVector(filtered_triangles_main)
mesh_main_surf = o3d.geometry.TriangleMesh(vertices_main_surf, filtered_triangles_main)
mesh_main_surf.compute_vertex_normals()
'''
# Create a mesh from the filtered triangles for side wall surface
vertices_side_wall = o3d.utility.Vector3dVector(np.asarray(Side_wall_points_pcd.points))
filtered_triangles_side_wall = o3d.utility.Vector3iVector(filtered_triangles_side_wall)
mesh_side_wall = o3d.geometry.TriangleMesh(vertices_side_wall, filtered_triangles_side_wall)
mesh_side_wall.compute_vertex_normals()
'''



# Convert combined points to Open3D PointCloud
combined_pcd = o3d.geometry.PointCloud()
combined_pcd.points = o3d.utility.Vector3dVector(combined_points_step3)

# Compute normals for the combined point cloud
combined_pcd.estimate_normals()

# Smooth the mesh
number_of_iterations_pass = int(params['smooth_iterations'])
mesh_main_surf = mesh_main_surf.filter_smooth_simple(number_of_iterations=number_of_iterations_pass)
mesh_main_surf.compute_vertex_normals()

# Save the mesh to an .stl file
#output_path = "E:\\Fiverr\\charlesheld\\output_mesh_1.stl"
#o3d.io.write_triangle_mesh(output_path, mesh_main_surf)


# Visualize the result
#o3d.visualization.draw_geometries([mesh_main_surf])

#-----------------------------------------------------------------------------------------------------------------------------------------
# Project the boundary points onto the new plane
#projected_boundary_points = combined_points_step3.copy()
projected_boundary_points = np.asarray(mesh_main_surf.vertices).copy()

for i, point in enumerate(projected_boundary_points):
    vector_from_point_to_plane = point - new_plane_center
    distance_to_plane = np.dot(vector_from_point_to_plane, new_plane_normal)
    projected_boundary_points[i] = point - distance_to_plane * new_plane_normal
    
# Create Open3D point clouds for visualization
projected_boundary_pcd = o3d.geometry.PointCloud()
projected_boundary_pcd.points = o3d.utility.Vector3dVector(projected_boundary_points)

#Combine for bottom surface points
bottom_surface_points = np.vstack((projected_boundary_points))
bottom_surface_points_pcd = o3d.geometry.PointCloud()
bottom_surface_points_pcd.points = o3d.utility.Vector3dVector(bottom_surface_points)


tri_2 = Delaunay(bottom_surface_points[:, :2])
Bottom_surface_triangles = tri_2.simplices

filtered_triangles_bottom_surf = filter_large_faces(Bottom_surface_triangles, np.asarray(bottom_surface_points_pcd.points), max_edge_length)

# Create a mesh from the filtered triangles for bottom surface
vertices_bottom_surface = o3d.utility.Vector3dVector(np.asarray(bottom_surface_points_pcd.points))
filtered_triangles_bottom_surf = o3d.utility.Vector3iVector(filtered_triangles_bottom_surf)
mesh_bottom_surface = o3d.geometry.TriangleMesh(vertices_bottom_surface, filtered_triangles_bottom_surf)
mesh_bottom_surface.compute_vertex_normals()

#o3d.visualization.draw_geometries([mesh_main_surf, mesh_bottom_surface])
#-----------------------------------------------------------------------------------------------------------------------------------------
# Get the vertices from the main surface
vertices = np.asarray(mesh_main_surf.vertices)

# Find the minimum Z value of the bottom surface plane
bottom_surface_z = np.min(np.asarray(mesh_bottom_surface.vertices)[:, 2])

# Create a new set of vertices by moving them towards the bottom plane in the Z direction only
new_vertices = []
for vertex in vertices:
    # Move the vertex downwards in the Z direction
    new_z = max(vertex[2] - np.abs(vertex[2] - bottom_surface_z), bottom_surface_z)
    new_vertex = np.array([vertex[0], vertex[1], new_z])
    new_vertices.append(new_vertex)

# Convert to a numpy array
new_vertices = np.array(new_vertices)

# Combine original and new vertices
combined_vertices = np.vstack((vertices, new_vertices))

#-----------------------------------------------------------------------------------------------------------------------------------------
# Create faces for the new mesh as before
faces = []
faces.extend(np.asarray(mesh_main_surf.triangles))

# Add side faces to create a closed solid
num_vertices = len(vertices)
for tri in mesh_main_surf.triangles:
    i, j, k = tri
    faces.append([i, j, j + num_vertices])
    faces.append([i, j + num_vertices, i + num_vertices])
    faces.append([j, k, k + num_vertices])
    faces.append([j, k + num_vertices, j + num_vertices])
    faces.append([k, i, i + num_vertices])
    faces.append([k, i + num_vertices, k + num_vertices])

# Create the new mesh
new_mesh = o3d.geometry.TriangleMesh()
new_mesh.vertices = o3d.utility.Vector3dVector(combined_vertices)
new_mesh.triangles = o3d.utility.Vector3iVector(faces)

# Optionally, compute vertex normals for the new mesh
new_mesh.compute_vertex_normals()

# Combine the new mesh with the bottom surface
combined_mesh = new_mesh + mesh_bottom_surface


# Save the new mesh with fillet
# Get the filename with extension
filename_with_extension = os.path.basename(ply_path)
input_filename, _ = os.path.splitext(filename_with_extension)

input_file_dir = os.path.dirname(ply_path)
input_filename_process = 'Processing_' + input_filename + '.stl'
stl_mid_process_file = os.path.join(input_file_dir, input_filename_process)
o3d.io.write_triangle_mesh(stl_mid_process_file, combined_mesh)

# Visualize the result
#o3d.visualization.draw_geometries([combined_mesh])



#-----------------------------------------------------------------------------------------------------------------------------------------
# Define the paths
blender_path = params['blender_exe_file_path']  # Adjust path to Blender executable

# Derive the Blender script path from the input file's directory
input_dir = os.path.dirname(__file__)
blender_script_path = os.path.join(input_dir, 'process_mesh.py')  # Blender script should be in the same directory

# Generate the output file name by adding 'finishedfile_' to the input file name
output_filename = 'finishedfile_' + input_filename + '.stl'
output_file = params['output_file_path']

# Your existing code to handle mesh processing

# Step to run Blender script
def run_blender_script():
    # Build the command
    command = [
        blender_path,
        '--background',  # Run Blender in background mode
        '--python', blender_script_path,  # Path to the Blender script
        '--',  # Separator for arguments
        stl_mid_process_file,  # Input file path
        output_file  # Output file path
    ]

    # Execute the command
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Print output and errors
    print("Blender Output:")
    print(result.stdout)
    print("Blender Errors:")
    print(result.stderr)

# Call the function to run the Blender script
run_blender_script()

if os.path.exists(output_ply_sub_file):
    os.remove(output_ply_sub_file)
else:
    print(f"{output_ply_sub_file} does not exist.")

if os.path.exists(stl_mid_process_file):
    os.remove(stl_mid_process_file)
else:
    print(f"{stl_mid_process_file} does not exist.")

