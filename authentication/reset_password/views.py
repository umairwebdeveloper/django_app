from django.views.generic import TemplateView
from django.conf import settings
from _keenthemes.__init__ import KTLayout
from _keenthemes.libs.theme import KTTheme
from django.contrib import messages
from django.shortcuts import redirect
import uuid
from authentication.models import ResetPasswordToken
from authentication.reset_password.helpers import send_reset_password_mail
from django.contrib.auth import get_user_model

User = get_user_model()


class AuthResetPasswordView(TemplateView):
    template_name = "pages/auth/reset-password.html"

    def get_context_data(self, **kwargs):
        # Call the base implementation first to get a context
        context = super().get_context_data(**kwargs)

        # A function to init the global layout. It is defined in _keenthemes/__init__.py file
        context = KTLayout.init(context)

        KTTheme.addJavascriptFile(
            "js/custom/authentication/reset-password/reset-password.js"
        )

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
        # Check if email exists
        if not User.objects.filter(email=email).first():
            messages.error(request, "Not email found with this email.")
            return redirect("auth:reset-password")

        # Check if token exists for this email and if it is expired or not
        if email is not None:
            user_obj = User.objects.get(email=email)
            token = str(uuid.uuid4())
            reset_password_obj = ResetPasswordToken.objects.get(user=user_obj)
            reset_password_obj.token = token
            reset_password_obj.save()
            send_reset_password_mail(user_obj.email, token)
            messages.success(request, "An email is sent. Please check your inbox.")
            return redirect("auth:reset-password")
        else:
            messages.error(request, "Invalid email")
            return redirect("auth:reset-password")
