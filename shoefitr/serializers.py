from django.contrib.auth import get_user_model
from django.utils.encoding import force_str
from django.utils.http import urlsafe_base64_decode
from rest_framework import serializers
from rest_framework.exceptions import AuthenticationFailed
from rest_framework_simplejwt.tokens import RefreshToken, TokenError

from .tokens import gui_button_token
from django.contrib.auth.models import update_last_login

User = get_user_model()


class IframeSerializer(serializers.Serializer):
    token = serializers.CharField(min_length=1, write_only=True)
    uidb64 = serializers.CharField(min_length=1, write_only=True)

    class Meta:
        fields = ["token", "uidb64"]

    def validate(self, attrs):
        token = attrs.get("token")
        uidb64 = attrs.get("uidb64")

        id = force_str(urlsafe_base64_decode(uidb64))
        user = User.objects.get(id=id)
        if not gui_button_token.check_token(user, token):
            raise AuthenticationFailed("The link is invalid or expired")

        return super().validate(attrs)
