from django.contrib.auth import logout
from django.shortcuts import redirect
from django.views.generic import TemplateView


def index(request):
    if request.user:
        return redirect("dashboards:dashboard")    
    return redirect("auth:signin")
