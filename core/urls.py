from django.urls import path
from .views import VideoUploadView, TestApi

urlpatterns = [
    path('api/upload-video/', VideoUploadView.as_view(), name='upload-video'),
    path('api/test/', TestApi.as_view(), name='test')
]
