from django import forms
from shoefitr1.models import User
from django.contrib.auth.forms import UserCreationForm


class UserRegisterForm(UserCreationForm):

    class Meta:
        model = User
        fields = ['username', 'password1', 'password2']
