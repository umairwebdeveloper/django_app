import threading

from django.contrib.auth.tokens import PasswordResetTokenGenerator
from django.core.mail import EmailMessage, EmailMultiAlternatives
from django.contrib.auth.hashers import make_password, check_password
import six


class EmailThread(threading.Thread):
    def __init__(self, email):
        self.email = email
        threading.Thread.__init__(self)

    def run(self):
        self.email.send()


class Util:
    @staticmethod
    def send_email(data):
        mail = EmailMultiAlternatives(
            subject=data["email_subject"],
            body=data["email_body"],
            to=data["to_email"],
            from_email="Shoefitr <no_reply@shoefitr.io>",
        )
        if data.get("html", None):
            mail.attach_alternative(data["email_body"], "text/html")

        EmailThread(mail).start()


class AccountActivationTokenGenerator(PasswordResetTokenGenerator):
    def _make_hash_value(self, user, timestamp):
        return (
            six.text_type(user.pk)
            + six.text_type(timestamp)
            + six.text_type(user.profile.email_confirmed)
        )


class GUIButtonTokenGenerator:
    def check_token(self, user, token):
        email_field = user.get_email_field_name()
        email = getattr(user, email_field, "") or ""
        username = user.username or ""
        return check_password(f"{user.pk}{username}{email}", token)

    def make_token(self, user):
        email_field = user.get_email_field_name()
        email = getattr(user, email_field, "") or ""
        username = user.username or ""
        return make_password(f"{user.pk}{username}{email}")

class PasswordResetToken(PasswordResetTokenGenerator):
    def _make_hash_value(self, user, timestamp):
        login_timestamp = (
            ""
            if user.last_login is None
            else user.last_login.replace(microsecond=0, tzinfo=None)
        )
        updated_timestamp = (
            ""
            if user.updated_on is None
            else user.updated_on.replace(microsecond=0, tzinfo=None)
        )
        email_field = user.get_email_field_name()
        email = getattr(user, email_field, "") or ""
        return f"{user.pk}{user.password}{login_timestamp}{timestamp}{updated_timestamp}{email}"


account_activation_token = AccountActivationTokenGenerator()
password_reset_token = PasswordResetToken()
gui_button_token = GUIButtonTokenGenerator()
