from django.urls import path
from .views import VideoUploadView, TestApi, CalculateSizeView

urlpatterns = [
    path('api/upload-video/', VideoUploadView.as_view(), name='upload-video'),
    path('api/test/', TestApi.as_view(), name='test'),
    path('api/calculate-size/', CalculateSizeView.as_view(), name='calculate-size'),
]
