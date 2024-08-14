from rest_framework.views import APIView
from rest_framework.response import Response
from io import BytesIO
from PIL import Image
import open3d as o3d
from .depth_to_surface import calculate_cloud_point_from_pic
from shoefitr.models import data
from django.core.files.base import ContentFile


class CalculateCloudPointView(APIView):
    
    def post(self, request, *args, **kwargs):
        # Getting the uploaded file
        uploaded_file = request.FILES['file']
        
        # Reading the image from the uploaded file
        image_data = uploaded_file.read()
        image = Image.open(BytesIO(image_data))
        
        # Calculating the cloud point from the image
        point_cloud = calculate_cloud_point_from_pic(image)
        
        # Saving the point cloud to a .ply file in memory
        new_image_name = 'cloud_point.ply'
        o3d.io.write_point_cloud(new_image_name, point_cloud)

        # Reading the .ply file into memory
        with open(new_image_name, 'rb') as f:
            ply_content = f.read()
        
        # Save the file to the Django model
        point_cloud_instance = data(model_name="test_insole")
        point_cloud_instance.file.save(new_image_name, ContentFile(ply_content))
        point_cloud_instance.save()
        
        # Returning a response
        return Response({"message": "Point cloud saved to the database"})