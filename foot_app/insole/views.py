from rest_framework.views import APIView
from rest_framework.response import Response
from io import BytesIO
from PIL import Image
import open3d as o3d
from .depth_to_surface import calculate_cloud_point_from_pic
from shoefitr.models import data
from django.core.files.base import ContentFile
import tempfile
import os

class CalculateCloudPointView(APIView):
    
    def post(self, request, *args, **kwargs):
        # Getting the uploaded file
        uploaded_file = request.FILES['file']

        # Reading the image from the uploaded file
        image = Image.open(uploaded_file)

        # Calculating the cloud point from the image
        point_cloud = calculate_cloud_point_from_pic(image)

        # Save point cloud to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".ply") as temp_file:
            temp_filename = temp_file.name
            o3d.io.write_point_cloud(temp_filename, point_cloud, write_ascii=True)
        
        # Read the file into a BytesIO object
        with open(temp_filename, 'rb') as f:
            ply_content = f.read()
        
        # Create ContentFile from the BytesIO object
        content_file = ContentFile(ply_content)

        # Clean up the temporary file
        os.remove(temp_filename)

        # Save the file directly to the Django model
        point_cloud_instance = data(model_name="test_insole")
        point_cloud_instance.file.save('cloud_point.ply', content_file)
        point_cloud_instance.save()

        # Returning a response
        return Response({"message": "Point cloud saved to the database"})