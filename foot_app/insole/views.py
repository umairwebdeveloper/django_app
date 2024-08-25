from rest_framework.views import APIView
from rest_framework.response import Response
from io import BytesIO
from PIL import Image
import open3d as o3d
from .depth_to_surface import calculate_cloud_point_from_pic
from shoefitr.models import data
from django.core.files.base import ContentFile
from rest_framework import status
import tempfile
import os

class CalculateCloudPointView(APIView):
    
    def post(self, request, *args, **kwargs):
        try:
            # Getting the uploaded file
            uploaded_file = request.FILES.get('file')
            if not uploaded_file:
                return Response({"error": "Please upload a valid image file."}, status=status.HTTP_400_BAD_REQUEST)

            # Reading the image from the uploaded file
            try:
                image = Image.open(uploaded_file)
            except Exception as e:
                return Response({"error": "The uploaded file could not be opened as an image. Please ensure it is a valid image file."}, status=status.HTTP_400_BAD_REQUEST)

            # Calculating the cloud point from the image
            try:
                point_cloud = calculate_cloud_point_from_pic(image)
            except Exception as e:
                return Response({"error": "We encountered an issue while processing the image to calculate the point cloud. Please try again with a different image."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            # Save point cloud to a temporary file
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".ply") as temp_file:
                    temp_filename = temp_file.name
                    o3d.io.write_point_cloud(temp_filename, point_cloud, write_ascii=True)
            except Exception as e:
                return Response({"error": "An error occurred while saving the point cloud. Please try again."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            # Read the file into a BytesIO object
            try:
                with open(temp_filename, 'rb') as f:
                    ply_content = f.read()
            except Exception as e:
                return Response({"error": "An error occurred while reading the point cloud data. Please try again."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            # Create ContentFile from the BytesIO object
            content_file = ContentFile(ply_content)

            # Clean up the temporary file
            try:
                os.remove(temp_filename)
            except Exception as e:
                # Log the error but don't fail the response
                print(f"Failed to delete temporary file: {str(e)}")

            # Save the file directly to the Django model
            try:
                point_cloud_instance = data(model_name="test_insole")
                point_cloud_instance.file.save('cloud_point.ply', content_file)
                point_cloud_instance.save()
            except Exception as e:
                return Response({"error": "We encountered an issue while saving the point cloud to our database. Please try again later."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            # Returning a success response
            return Response({"message": "Image processed successfully."}, status=status.HTTP_200_OK)

        except Exception as e:
            # Catch any other unforeseen errors
            return Response({"error": "An unexpected error occurred. Please try again later."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
