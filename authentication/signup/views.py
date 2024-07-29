from django.views.generic import TemplateView
from django.conf import settings
from _keenthemes.__init__ import KTLayout
from _keenthemes.libs.theme import KTTheme
from django.shortcuts import redirect
from django.contrib import messages
from authentication.models import ResetPasswordToken
from django.contrib.auth import get_user_model
from django.contrib.auth.models import Group


User = get_user_model()
"""
This file is a view controller for multiple pages as a module.
Here you can override the page view layout.
Refer to urls.py file for more pages.


"""


class AuthSignupView(TemplateView):
    template_name = "pages/auth/signup.html"

    def post(self, request, *args, **kwargs):
        # Handle post request here
        email = request.POST.get("email")
        password = request.POST.get("password")
        confirm_password = request.POST.get("confirm-password")
        username = email.split("@")[0]
        if (
            not User.objects.filter(email=email).exists()
            and not User.objects.filter(username=username).exists()
        ):
            if password == confirm_password:
                if len(password) < 8:
                    messages.error(request, "Password must be at least 8 characters")
                    return redirect("auth:signup")
                if len(password) > 20:
                    messages.error(request, "Password must be at most 20 characters")
                    return redirect("auth:signup")
                user = User.objects.create_user(username, email, password)
                
                reset_password_obj = ResetPasswordToken.objects.create(user=user)
                reset_password_obj.save()
                messages.success(request, "Account created successfully, please login.")
                return redirect("auth:signin")
            else:
                messages.error(request, "Passwords do not match")
                return redirect("auth:signup")
        else:
            messages.error(request, "Email already exists")
            return redirect("auth:signup")

    def get_context_data(self, **kwargs):
        # Call the base implementation first to get a context
        context = super().get_context_data(**kwargs)

        # A function to init the global layout. It is defined in _keenthemes/__init__.py file
        context = KTLayout.init(context)

        KTTheme.addJavascriptFile("js/custom/authentication/sign-up/general.js")

        # Define the layout for this module
        # _templates/layout/auth.html
        context.update(
            {
                "layout": KTTheme.setLayout("auth.html", context),
            }
        )

        return context
