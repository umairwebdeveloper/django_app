from django.views.generic import TemplateView
from django.conf import settings
from _keenthemes.__init__ import KTLayout
from _keenthemes.libs.theme import KTTheme
from django.contrib import messages
from django.shortcuts import redirect, render

# import uuid
from authentication.models import ResetPasswordToken

# from authentication.reset_password.helpers import send_reset_password_mail
from django.contrib.auth import get_user_model

User = get_user_model()


class AuthNewPasswordView(TemplateView):
    template_name = "pages/auth/new-password.html"

    def get_context_data(self, token, **kwargs):
        # Call the base implementation first to get a context
        context = super().get_context_data(**kwargs)

        # A function to init the global layout. It is defined in _keenthemes/__init__.py file
        context = KTLayout.init(context)

        KTTheme.addJavascriptFile(
            "js/custom/authentication/reset-password/new-password.js"
        )
        reset_obj = ResetPasswordToken.objects.filter(token=token).first()

        # Define the layout for this module
        # _templates/layout/auth.html
        context.update(
            {
                "layout": KTTheme.setLayout("auth.html", context),
                "user_id": reset_obj.user.id,
            }
        )

        return context

    def post(self, request, token, *args, **kwargs):
        try:
            if request.method == "POST":
                new_password = request.POST.get("password")
                confirm_password = request.POST.get("confirm-password")
                user_id = request.POST.get("user_id")

                if user_id is None:
                    messages.error(request, "No user id found.")
                    return redirect(f"/new-password/{token}/")

                if not 8 <= len(new_password) <= 20:
                    messages.error(
                        request, "Password must be between 8 and 20 characters."
                    )
                    return redirect(f"/new-password/{token}/")

                if new_password != confirm_password:
                    messages.error(
                        request, "Password and confirm password does not match."
                    )
                    return redirect(f"/new-password/{token}/")

                user_obj = User.objects.get(id=user_id)
                user_obj.set_password(new_password)
                user_obj.save()
                messages.success(
                    request, "Password changed successfully. Please login."
                )
                return redirect("auth:signin")

        except Exception as e:
            messages.error(request, "Something went wrong. Please try again.")
            print(e)

        return redirect(f"/new-password/{token}/")
