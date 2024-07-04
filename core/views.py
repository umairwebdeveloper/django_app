from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from .models import Video
from .serializers import VideoSerializer

class VideoUploadView(APIView):
    def post(self, request, *args, **kwargs):
        file_serializer = VideoSerializer(data=request.data)
        if file_serializer.is_valid():
            file_serializer.save()
            return Response(file_serializer.data, status=status.HTTP_201_CREATED)
        else:
            return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class TestApi(APIView):
    def get(self, request, format=None):
        return Response({"name": "hello world"})