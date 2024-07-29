from django.db import models
from djstripe.models import Customer, Subscription
from django.contrib.auth.models import AbstractUser, BaseUserManager, PermissionsMixin
from django.dispatch import receiver
from django.db.models.signals import post_save
# from django.contrib.postgres.fields import ArrayField


class UserManager(BaseUserManager):
    def create_user(self, username, password=None):
        if username is None:
            raise TypeError("Users should have a Username")

        user = self.model(username=username)
        user.set_password(password)
        user.save()
        return user

    def create_superuser(self, email, password=None):
        if password is None:
            raise TypeError("Password should not be none")

        user = self.create_user(email, password)
        user.is_superuser = True
        user.is_staff = True
        user.save()
        return user


AUTH_PROVIDERS = {
    "facebook": "facebook",
    "google": "google",
    "twitter": "twitter",
    "email": "email",
}


class User(AbstractUser):
    customer = models.ForeignKey(
        Customer,
        related_name="user_customers",
        null=True,
        blank=True,
        default=None,
        on_delete=models.SET_NULL,
    )
    subscription = models.ForeignKey(
        Subscription,
        related_name="user_subscriptions",
        null=True,
        blank=True,
        default=None,
        on_delete=models.SET_NULL,
    )
    avatar = models.ImageField(
        upload_to="avatars/",
        null=True,
        blank=True,
    )
    bio = models.TextField(null=True, blank=True)
    is_verified = models.BooleanField(default=False, null=True)
    auth_provider = models.CharField(
        max_length=255, blank=False, null=True, default=AUTH_PROVIDERS.get("email")
    )
    can_access_file = models.BooleanField(default=True, null=True)
    # allowed_domains = ArrayField(
    #     models.CharField(max_length=100, default="", blank=True),
    #     default=list,
    #     blank=True,
    #     null=True,
    # )
    is_admin_portal = models.BooleanField(default=False, null=True)
    is_shop_admin = models.BooleanField(default=False, null=True)
    is_customer = models.BooleanField(default=False, null=True)
    is_client = models.BooleanField(default=True, null=True)
    is_only_measurements = models.BooleanField(default=False, null=True)
    client = models.ForeignKey(
        "self",
        related_name="user_clients",
        null=True,
        blank=True,
        default=None,
        on_delete=models.SET_NULL,
    )
    paidShoesApis = models.IntegerField(default=0)
    created_on = models.DateTimeField(auto_now_add=True, null=True)
    updated_on = models.DateTimeField(auto_now=True, null=True)

    def __str__(self):
        return self.username

    class Meta:
        ordering = ["-id"]


class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    forget_password_token = models.CharField(max_length=100, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.user.username


# Create your models here.
class ResetPasswordToken(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    token = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.user.username


# when creating user, set username to email
@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        instance.username = instance.email
        instance.save()
        Profile.objects.create(user=instance)
