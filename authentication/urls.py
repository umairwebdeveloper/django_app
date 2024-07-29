from django.urls import path
from django.conf import settings
from authentication.signin.views import AuthSigninView
from authentication.signup.views import AuthSignupView
from authentication.reset_password.views import AuthResetPasswordView
from authentication.new_password.views import AuthNewPasswordView
from authentication.signout.views import AuthSignoutView
from authentication.home.views import index

app_name = "auth"

urlpatterns = [
    path(
        "",
        index,
        name="home",
    ),
    path(
        "signin/",
        AuthSigninView.as_view(template_name="pages/auth/signin.html"),
        name="signin",
    ),
    path(
        "wi3bit/login/",
        AuthSigninView.as_view(template_name="pages/wi3bit/auth/login.html"),
        name="wi3bit_login",
    ),
    path(
        "signup/",
        AuthSignupView.as_view(template_name="pages/auth/signup.html"),
        name="signup",
    ),
    path("signout/", AuthSignoutView.as_view(), name="signout"),
    path(
        "reset-password/",
        AuthResetPasswordView.as_view(template_name="pages/auth/reset-password.html"),
        name="reset-password",
    ),
    path(
        "new-password/<token>/",
        AuthNewPasswordView.as_view(template_name="pages/auth/new-password.html"),
        name="new-password",
    ),
]
