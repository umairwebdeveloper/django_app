from django.urls import path
from . import views
from foot_app.views import calculation, calculation_ar, calculation_only_measurements
from django.conf import settings
from django.conf.urls.static import static
from rest_framework.authtoken.views import obtain_auth_token

urlpatterns = [
    # url(r'shoedata/(\w+)/', views.uploadshoedata, name='uploadshoedata'),
    # path("apidata/<shopID>/<userID>/", views.returnjsondata, name="returnjsondata"),
    # url(r'^requestpictures/$', views.Get_shoes_List.as_view(), name='requestpictures'),
    # url(r'usersapi/', views.usersListApi.as_view(), name='usersapi'),
    path("save/", views.save, name="save"),
    path("update_server/", views.update_server, name="update_server"),
    path("s/", views.scan, name="s"),
    path("confirm-scan/", views.confirm_scan, name="confirm-scan"),
    path("scan/", views.instructions, name="scan"),
    path("scan-ar/", views.scan_ar, name="scan_ar"),
    path("scan-button/", views.iframe_page, name="scan_button"),
    path("info/", views.info, name="info"),
    # path("apiadvice/", views.apiadvice, name="apiadvice"),
    path("calculation/", calculation, name="calculation"),
    path("calculation-only-measurements/", calculation_only_measurements, name="calculation_measurements"),
    path("calculation-ar/", calculation_ar, name="calculation_ar"),
    path("hello/", views.HelloView.as_view(), name="hello"),
    path("auth_token", obtain_auth_token, name="api_token_auth"),
    path("test-image", views.test_image, name="test-image"),
    path("new/", views.new_page, name="new-page"),
    path("8thwall-response/", views.ethwall_response, name="ethwall_response"),
    path('match/ids/', views.MatchUserIdShopOwner.as_view(), name='match-userid-shopowner'),
    path('shop/ids/', views.ShopIDListView.as_view(), name='shop-id-list'),
    path('shop/model-names/<str:shopid>/', views.ModelNamesListView.as_view(), name='model-names-list'),
]
