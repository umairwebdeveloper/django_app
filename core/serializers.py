from rest_framework import serializers
from .models import Video

class VideoSerializer(serializers.ModelSerializer):
    class Meta:
        model = Video
        fields = ('video', 'uploaded_at')

class FileUploadSerializer(serializers.Serializer):
    file = serializers.FileField()