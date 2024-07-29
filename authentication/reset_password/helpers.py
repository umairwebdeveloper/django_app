from django.core.mail import EmailMultiAlternatives
from django.template.loader import render_to_string
from django.utils.html import strip_tags
from django.conf import settings

def send_reset_password_mail(email, token):
    subject = "Reset Your Password"

    # Render the HTML template for the email
    html_message = render_to_string("pages/auth/email.html", {"token": token})

    # Extract the text content from HTML for the plain text version of the email
    text_content = strip_tags(html_message)

    email_from = settings.EMAIL_HOST_USER
    recipient_list = [email]

    # Create an EmailMultiAlternatives object to include both HTML and plain text versions of the email
    msg = EmailMultiAlternatives(subject, text_content, email_from, recipient_list)
    msg.attach_alternative(html_message, "text/html")

    # Send the email
    msg.send()
    return True
