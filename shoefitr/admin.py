from django.conf import settings
from django.contrib import admin

from .snippets import generate_presigned_url
from .models import (
    Reference,
    Shop,
    Shoes,
    Settings,
    data,
    margin,
)
from django.utils.html import format_html




class DataAdmin(admin.ModelAdmin):
    list_display = ["get_shop", "file", "download_link", "created_on"]
    
    def get_shop(self, obj):
        if obj.model_name == "test_insole":
            return "Test Insole"
        return obj.shop.shopOwner.username if obj.shop and obj.shop.shopOwner else obj.shop.id if obj.shop else "-"

    get_shop.short_description = "Shop"
    
    def download_link(self, obj):   
        if obj.file:
            file_key = settings.AWS_PRIVATE_MEDIA_LOCATION + "/" + obj.file.name
            presigned_url = generate_presigned_url(
                file_key
            )  # Generate presigned url for the file
            if presigned_url:
                return format_html('<a href="{}" download>Download</a>', presigned_url)
            return "Error generating link"
        return "-"

    download_link.short_description = "Download Link"




admin.site.register(Shoes)
admin.site.register(Settings)
admin.site.register(data, DataAdmin)
admin.site.register(Shop)
admin.site.register(Reference)
admin.site.register(margin)
    
