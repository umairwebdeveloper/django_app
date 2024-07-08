from django.contrib import admin
from .models import Video


@admin.register(Video)
class VideoAdmin(admin.ModelAdmin):
    list_display = ('id', 'video', 'uploaded_at')
    search_fields = ('video',) 

