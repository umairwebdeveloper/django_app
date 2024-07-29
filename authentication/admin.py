from django.contrib import admin
from authentication.models import ResetPasswordToken, User
# Register your models here.

admin.site.register(ResetPasswordToken)
admin.site.register(User)
# unregister old user and register new user abstract

# from django.contrib.auth.admin import UserAdmin
# from .models import Profile, 
