from django.views.generic import TemplateView
from django.conf import settings
from _keenthemes.__init__ import KTLayout
from _keenthemes.libs.theme import KTTheme
from django.contrib.auth import authenticate, login
from django.shortcuts import redirect
from django.contrib import messages
"""
This file is a view controller for multiple pages as a module.
Here you can override the page view layout.
Refer to urls.py file for more pages.
"""


class AuthSigninView(TemplateView):
    template_name = "pages/auth/signin.html"

    def get_context_data(self, **kwargs):
        # Call the base implementation first to get a context
        context = super().get_context_data(**kwargs)

        # A function to init the global layout. It is defined in _keenthemes/__init__.py file
        context = KTLayout.init(context)

        KTTheme.addJavascriptFile("js/custom/authentication/sign-in/general.js")

        # Define the layout for this module
        # _templates/layout/auth.html
        context.update(
            {
                "layout": KTTheme.setLayout("auth.html", context),
            }
        )

        return context
    
    def post(self, request, *args, **kwargs):
        # Handle post request here
        email = request.POST.get("email")
        password = request.POST.get("password")
        username = email.split("@")[0]
        user = authenticate(request, username=email, password=password)
        if user is not None:
            login(request, user)
            messages.success(request, "Login successful")
            return redirect("dashboards:dashboard")
        else:
            messages.error(request, "Invalid email or password")
            return redirect("auth:signin")
        

        
