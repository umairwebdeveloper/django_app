from django.db import models
from django_app.storage_backends import PrivateMediaStorage
from django.contrib.postgres.fields import ArrayField


from django.contrib.auth import get_user_model

User = get_user_model()


class PostpaidPlans(models.Model):
    name = models.CharField(max_length=100, null=True)
    pricePerApi = models.FloatField(null=True, blank=True)
    desc = models.CharField(max_length=300, null=True, blank=True)
    subDesc = models.CharField(max_length=300, default="", blank=True)

    def __str__(self):
        return self.name

    class Meta:
        ordering = ["-id"]
        db_table = "dashboards_postpaidplans"
        verbose_name_plural = "post_paid_plans"


class Profile(models.Model):
    full_name = models.CharField(max_length=100, null=True, blank=True)
    bio = models.TextField(null=True, blank=True)
    contact = models.CharField(max_length=15, null=True, blank=True)
    avatar = models.ImageField(upload_to="pictures/avatars/", null=True, default=None)
    created_on = models.DateTimeField(auto_now_add=True, null=True)
    updated_on = models.DateTimeField(auto_now=True, null=True)

    class Meta:
        ordering = ["-created_on"]
        verbose_name_plural = "profiles"
        db_table = "dashboards_profile"


class Shop(models.Model):
    shopOwner = models.OneToOneField(
        get_user_model(),
        on_delete=models.SET_NULL,
        null=True,
    )
    logo = models.ImageField(
        storage=PrivateMediaStorage(),
        upload_to="shop_data/",
        null=True,
        blank=True,
    )

    title = models.CharField(max_length=30, null=True, blank=True)
    subTitle = models.CharField(max_length=30, null=True, blank=True)
    currency = models.CharField(max_length=50, null=True, blank=True)
    minOrderAmount = models.DecimalField(
        max_digits=10, decimal_places=0, null=True, blank=True
    )
    walletCurrencyRatio = models.CharField(max_length=10, null=True, blank=True)
    taxClass = models.CharField(max_length=50, null=True, blank=True)
    signupPoints = models.CharField(max_length=10, null=True, blank=True)
    shippingClass = models.CharField(max_length=50, null=True, blank=True)
    metaTitle = models.CharField(max_length=30, null=True, blank=True)
    metaDesc = models.CharField(max_length=200, null=True, blank=True)
    metaTags = models.CharField(max_length=200, null=True, blank=True)
    canonicalUrl = models.URLField(null=True, blank=True)
    ogTitle = models.CharField(max_length=30, null=True, blank=True)
    ogDesc = models.CharField(max_length=200, null=True, blank=True)
    ogImage = models.ImageField(
        storage=PrivateMediaStorage(),
        upload_to="shop_data/",
        null=True,
        default=None,
        blank=True,
    )

    twitterHandle = models.CharField(max_length=30, null=True, blank=True)
    twitterCardType = models.CharField(max_length=30, null=True, blank=True)
    lat = models.IntegerField(null=True, blank=True)
    lng = models.IntegerField(null=True, blank=True)
    state = models.CharField(max_length=30, null=True, blank=True)
    country = models.CharField(max_length=30, null=True, blank=True)
    city = models.CharField(max_length=30, null=True, blank=True)
    zip = models.IntegerField(null=True, blank=True)
    formattedAddress = models.CharField(max_length=200, null=True, blank=True)
    formattedAddress2 = models.CharField(max_length=200, null=True, blank=True)
    contact = models.CharField(max_length=30, null=True, blank=True)
    email = models.EmailField(null=True, blank=True)
    website = models.CharField(max_length=30, null=True, blank=True)
    created_on = models.DateTimeField(auto_now_add=True, null=True)
    updated_on = models.DateTimeField(auto_now=True, null=True)
    free_api = models.IntegerField(default=100, null=True, blank=True)
    web_api_token = models.CharField(max_length=50, null=True, blank=True)

    def __str__(self):
        return self.shopOwner.username if self.shopOwner else str(self.id)

    class Meta:
        ordering = ["-created_on"]
        verbose_name_plural = "shops"
        db_table = "dashboards_shop"


class Shoes(models.Model):
    shop = models.ForeignKey(
        Shop,
        related_name="shoes_shops",
        on_delete=models.RESTRICT,
        null=True,
        default=None,
    )
    userid = models.CharField(max_length=45, null=True, default=None)
    modelid = models.CharField(max_length=45, null=True, default=None)
    length = models.CharField(max_length=45, null=True, default=None)
    width = models.CharField(max_length=45, null=True, default=None)
    size_eu = models.CharField(max_length=45, null=True, default=None)
    width_advice = models.CharField(max_length=45, null=True, default=None)  # size_uk
    ball_advice = models.CharField(max_length=45, null=True, default=None)
    instep_advice = models.CharField(max_length=45, null=True, default=None)
    model_name = models.CharField(max_length=45, null=True, default=None)  # size_us
    picture = models.ImageField(upload_to="pictures/", null=True, default=None)
    correct_size_found = models.BooleanField(default=False)
    shoespair = models.CharField(max_length=45, null=True, default=None)
    left_length = models.CharField(max_length=45, null=True, default=None)
    right_length = models.CharField(max_length=45, null=True, default=None)
    left_width = models.CharField(max_length=45, null=True, default=None)
    right_width = models.CharField(max_length=45, null=True, default=None)
    left_ball = models.CharField(max_length=45, null=True, default=None)
    right_ball = models.CharField(max_length=45, null=True, default=None)
    left_instep = models.CharField(max_length=45, null=True, default=None)
    right_instep = models.CharField(max_length=45, null=True, default=None)
    reference = models.ForeignKey(
        "Reference", null=True, blank=True, default=None, on_delete=models.SET_NULL
    )
    created_on = models.DateTimeField(auto_now_add=True, null=True)
    updated_on = models.DateTimeField(auto_now=True, null=True)

    def __str__(self):
        return str(self.modelid)

    class Meta:
        ordering = ["-created_on"]
        verbose_name_plural = "shoes"
        db_table = "dashboards_shoes"


class data(models.Model):
    file = models.FileField(
        storage=PrivateMediaStorage(),
        upload_to="last_files/",
        max_length=255,
        null=True,
        default=None,
    )
    shop = models.OneToOneField(
        Shop,
        related_name="data_shop",
        on_delete=models.SET_NULL,
        null=True,
        default=None,
    )
    length_up_margin = models.CharField(max_length=10, null=True, default=None)
    width_up_margin = models.CharField(max_length=10, null=True, default=None)
    length_down_margin = models.CharField(max_length=10, null=True, default=None)
    width_down_margin = models.CharField(max_length=10, null=True, default=None)
    ball_up_margin = models.CharField(max_length=10, null=True, default=None)
    instep_up_margin = models.CharField(max_length=10, null=True, default=None)
    ball_down_margin = models.CharField(max_length=10, null=True, default=None)
    instep_down_margin = models.CharField(max_length=10, null=True, default=None)
    model_name = models.CharField(max_length=255, null=True, default=None)
    created_on = models.DateTimeField(auto_now_add=True, null=True)
    updated_on = models.DateTimeField(auto_now=True, null=True)

    def __str__(self):
        return str(self.shop.shopOwner) if self.shop else str(self.id)

    class Meta:
        ordering = ["-created_on"]
        verbose_name_plural = "data"
        db_table = "dashboards_data"


class margin(models.Model):
    file_data = models.ForeignKey(
        data, on_delete=models.CASCADE, related_name="margin_files"
    )
    length_up_margin = models.CharField(max_length=10, null=True, default=None)
    width_up_margin = models.CharField(max_length=10, null=True, default=None)
    length_down_margin = models.CharField(max_length=10, null=True, default=None)
    width_down_margin = models.CharField(max_length=10, null=True, default=None)
    ball_up_margin = models.CharField(max_length=10, null=True, default=None)
    instep_up_margin = models.CharField(max_length=10, null=True, default=None)
    ball_down_margin = models.CharField(max_length=10, null=True, default=None)
    instep_down_margin = models.CharField(max_length=10, null=True, default=None)
    margin_id = models.CharField(max_length=100, null=True, default=None)
    created_on = models.DateTimeField(auto_now_add=True, null=True)
    updated_on = models.DateTimeField(auto_now=True, null=True)

    def __str__(self):
        return (
            str(self.file_data.shop.shopOwner)
            if self.file_data.shop
            else str(self.file_data.id)
        )

    class Meta:
        ordering = ["-created_on"]
        verbose_name_plural = "margins"
        db_table = "dashboards_margin"


class Reference(models.Model):
    size = models.CharField(max_length=10, null=True, default=None)
    selection = models.CharField(max_length=50, null=True, default=None)
    region = models.CharField(max_length=50, null=True, default=None)
    created_on = models.DateTimeField(auto_now_add=True, null=True)
    updated_on = models.DateTimeField(auto_now=True, null=True)

    def __str__(self):
        return self.size if self.size is not None else "-"

    class Meta:
        ordering = ["-created_on"]
        verbose_name_plural = "references"
        db_table = "dashboards_reference"


class Device(models.Model):
    device_id = models.CharField(max_length=255)
    user_id = models.CharField(max_length=255)
    shoe_id = models.CharField(max_length=255)
    shoe_name = models.CharField(max_length=255)
    max_width = models.FloatField()
    max_height = models.FloatField()
    # point_cloud is array of array field, with inner array of 3 elements of float
    # point_cloud = ArrayField(ArrayField(models.FloatField(), size=3))
    created_on = models.DateTimeField(auto_now_add=True, null=True)
    updated_on = models.DateTimeField(auto_now=True, null=True)

    def __str__(self):
        return self.device_id if self.device_id is not None else "-"

    class Meta:
        ordering = ["-created_on"]
        verbose_name_plural = "devices"
        db_table = "dashboards_device"


class Settings(models.Model):
    scale = models.FloatField(default=4.25)
    created_on = models.DateTimeField(auto_now_add=True, null=True)
    updated_on = models.DateTimeField(auto_now=True, null=True)

    def __str__(self):
        return "Settings scale: " + str(self.scale)

    class Meta:
        verbose_name_plural = "Settings"
        verbose_name = "Settings"
        db_table = "dashboards_settings"
