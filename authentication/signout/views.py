from django.contrib.auth import logout
from django.shortcuts import redirect
from django.views.generic import TemplateView


class AuthSignoutView(TemplateView):
    def get(self, request):
        logout(request)
        return redirect("auth:signin")
