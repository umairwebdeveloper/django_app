from fastapi import FastAPI, File, UploadFile
import numpy as np
import uvicorn
from depth_to_surface import calculate_cloud_point_from_pic
from io import BytesIO
from PIL import Image
import open3d as o3d
app = FastAPI()


@app.post("/calculate_cloud_point")
async def calculate_cloud_point(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(BytesIO(image_data))
    point_cloud = calculate_cloud_point_from_pic(image)
    new_image_name = 'cloud_point.ply'
    o3d.io.write_point_cloud(new_image_name, point_cloud)
    # this is to visualize/show the cloud point file
    o3d.visualization.draw_geometries([point_cloud])
    # print("result saved")
    return "result saved"

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
