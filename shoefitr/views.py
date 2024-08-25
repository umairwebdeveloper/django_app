import json
import json as simplejson
import os
from builtins import str, super
from urllib.parse import urlparse
import numpy as np
import cv2
from datetime import datetime

import pandas as pd
from django.conf import settings
from django.contrib import messages
from django.contrib.auth import get_user_model
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import update_last_login
from django.http import HttpResponse, JsonResponse, HttpResponseRedirect
from django.urls import reverse
from django.shortcuts import redirect, render
from django.utils.encoding import force_str
from django.utils.http import urlsafe_base64_decode
from django.views.decorators.clickjacking import xframe_options_exempt
from django.views.decorators.csrf import csrf_exempt
from foot_app.snippets import detect_fun
from google.oauth2 import service_account
from rest_framework import serializers
from rest_framework.exceptions import AuthenticationFailed
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework_simplejwt.tokens import RefreshToken, TokenError
from rest_framework.decorators import api_view
from rest_framework import status
from .forms import shoesform, uploadshoedataForm
from .models import Shoes
from .tokens import gui_button_token
from utils.specific import detect_square, to_base64
from .snippets import get_model_names_from_file
User = get_user_model()


# @xframe_options_exempt
def iframe_page(request):
    token = request.GET.get("token")
    unb64 = request.GET.get("unb64")
    modelname = request.GET.get("modelname")
    token = token.replace(" ", "+")
    print(22, token, unb64)
    if not token or not unb64:
        return JsonResponse({"message": "Please provide token and uidb64"}, status=403)
    username = force_str(urlsafe_base64_decode(unb64))
    print(23, username)
    user = User.objects.filter(username=username).first()
    print(34, user)
    if not user:
        return JsonResponse({"message": "Token is invalid"}, status=403)
    if not gui_button_token.check_token(user, token):
        print(55, "not verified")
        raise AuthenticationFailed("The link is invalid or expired")
    host_url = request.META.get("HTTP_REFERER")
    print(345, host_url)
    if not host_url:
        return JsonResponse(
            {
                "message": "Unable to identify your site domain, please set 'referrerpolicy' to 'origin'."
            },
            status=403,
        )
    req_domain = urlparse(host_url).netloc.replace("www.", "")
    print(333333, req_domain)
    allowed_domains = user.allowed_domains + [
        "admin.shoefitr.io",
        "staging.admin.shoefitr.io",
        "testscan.shoefitr.io",
        "portal.shoefitr.io",
    ]
    if req_domain not in allowed_domains:
        # data = {
        #     "email_subject": "Someone tried to access Shoefitr web api with unauthorized domain.",
        #     "email_body": f"Shop: {user.title} <br> Request camee from: {reqDomain} <br> Allowed domain: {shopDomain}",
        #     "to_email": settings.ADMINS,
        #     "html": True,
        # }
        # Util.send_email(data)
        print(403, "Not allowed_domains", allowed_domains)
        response = JsonResponse(
            {
                "message": "This domain is not registered to shoefitr for your shop, please contact charles@shoefitr.io"
            },
            status=403,
        )
        response["Content-Security-Policy"] = "frame-ancestors 'none'"
        response["X-Frame-Options"] = "DENY"
        return response
    context = {"shopid": username, "userid": "12345", "modelname": modelname}
    response = render(request, "gui.html", context)
    response["Content-Security-Policy"] = f"frame-ancestors {host_url}"
    response["X-Frame-Options"] = f"ALLOW-FROM {host_url}"

    return response


class HelloView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        content = {"message": "Hello, World!"}
        return Response(content)


def confirm_scan(request):
    authenticated = False
    webshop_client_ids = [str(user) for user in User.objects.all()]
    try:
        shopid = request.GET["shopid"]
        userid = request.GET["userid"]
        name = request.GET["modelname"]
        redirecturl = request.GET["redirecturl"]
        if shopid == "admin":
            return HttpResponse(
                json.dumps(
                    {"Error": "Shop ID is not valid"},
                    default=str,
                    sort_keys=True,
                    indent=4,
                ),
                content_type="application/json",
            )
        elif shopid in webshop_client_ids:
            authenticated = True
        else:
            return HttpResponse(
                json.dumps(
                    {"Error": "Shop ID is not valid"},
                    default=str,
                    sort_keys=True,
                    indent=4,
                ),
                content_type="application/json",
            )
    except:
        return HttpResponse(
            json.dumps(
                {"Error": "URL parameters are not correct"},
                default=str,
                sort_keys=True,
                indent=4,
            ),
            content_type="application/json",
        )

    if authenticated:

        # data = jsondata(shopid)
        # print(size)
        # context = {'shopid':shopid, 'userid':userid, 'name':name, 'modelid':modelid, 'redirecturl':redirecturl, 'sizeeu':size['size_eu'],'sizeuk':size['size_uk'],'sizeus':size['size_us'],'length':size['length'],'width':size['width']}
        context = {
            "shopid": shopid,
            "userid": userid,
            "name": name,
            "redirecturl": redirecturl,
        }
        # context.update(data)
        return render(request, "index.html", context)


def scan(request):
    authenticated = False
    webshop_client_ids = [str(user) for user in User.objects.all()]
    try:
        shopid = request.GET["shopid"]
        userid = request.GET["userid"]
        name = request.GET["modelname"]
        redirecturl = request.GET["redirecturl"]
        if shopid == "admin":
            return HttpResponse(
                json.dumps(
                    {"Error": "Shop ID is not valid"},
                    default=str,
                    sort_keys=True,
                    indent=4,
                ),
                content_type="application/json",
            )
        elif shopid in webshop_client_ids:
            authenticated = True
        else:
            return HttpResponse(
                json.dumps(
                    {"Error": "Shop ID is not valid"},
                    default=str,
                    sort_keys=True,
                    indent=4,
                ),
                content_type="application/json",
            )
    except:
        return HttpResponse(
            json.dumps(
                {"Error": "URL parameters are not correct"},
                default=str,
                sort_keys=True,
                indent=4,
            ),
            content_type="application/json",
        )

    if authenticated:

        # data = jsondata(shopid)
        # print(size)
        # context = {'shopid':shopid, 'userid':userid, 'name':name, 'modelid':modelid, 'redirecturl':redirecturl, 'sizeeu':size['size_eu'],'sizeuk':size['size_uk'],'sizeus':size['size_us'],'length':size['length'],'width':size['width']}
        context = {
            "shopid": shopid,
            "userid": userid,
            "name": name,
            "redirecturl": redirecturl,
        }
        # context.update(data)
        # return render(request, "shoefitr/scan.html", context)
        return render(request, "index.html", context)


def instructions(request):
    authenticated = False
    webshop_client_ids = [str(user) for user in User.objects.all()]
    try:
        shopid = request.GET["shopid"]
        userid = request.GET["userid"]
        is_only_measurements = User.objects.get(username=shopid).is_only_measurements
        if is_only_measurements is True:
            name = "_"
            html_file = "measurements.html"
        else:
            html_file = "index.html"
            name = request.GET["modelname"]
        redirecturl = request.GET.get("redirecturl", "")
        marginid = request.GET.get("marginid", "")
        if shopid == "admin":
            return HttpResponse("Shop ID is not valid")
        if shopid in webshop_client_ids:
            authenticated = True
        else:
            return HttpResponse("Shop ID is not valid")
    except Exception as e:
        return HttpResponse(e.args[0] + ", URL parameters are not correct!")

    if authenticated:
        context = {
            "shopid": shopid,
            "userid": userid,
            "name": name,
            "redirecturl": redirecturl,
            "marginid": marginid,
        }
        return render(request, html_file, context)


def info(request):
    authenticated = False
    webshop_client_ids = [str(user) for user in User.objects.all()]
    try:
        shopid = request.GET["shopid"]
        userid = request.GET["userid"]
        name = request.GET["modelname"]
        redirecturl = request.GET["redirecturl"]
        if shopid == "admin":
            return HttpResponse("Shop ID is not valid")
        if shopid in webshop_client_ids:
            authenticated = True
        else:
            return HttpResponse("Shop ID is not valid")
    except:
        return HttpResponse("URL parameters are not correct!")

    if authenticated:
        context = {
            "shopid": shopid,
            "userid": userid,
            "name": name,
            "redirecturl": redirecturl,
        }
        return render(request, "shoefitr/instructions.html", context)


# @csrf_exempt
def save(request):
    shopid = request.POST.get("shopid", None)
    userid = request.POST.get("userid", None)
    modelid = request.POST.get("modelid", None)
    length = request.POST.get("length", None)
    width = request.POST.get("width", None)
    size = request.POST.get("size_eu", None)
    width_advice = request.POST.get("size_uk", None)
    model_name = request.POST.get("size_us", None)
    picture = request.POST.get("picture", None)
    shoespair = request.POST.get("shoespair", None)

    confirmation = "false"
    allusers = User.objects.all()
    allusers = allusers.exclude(is_superuser=True)
    for eachuser in allusers:
        if shopid == str(eachuser.username):
            Shoes.objects.create(
                shopid=shopid,
                userid=userid,
                modelid=modelid,
                length=length,
                width=width,
                size_eu=size,
                width_advice=width_advice,
                model_name=model_name,
                picture=picture,
                shoespair=shoespair,
            )
            confirmation = "true"

    data = {"saved": confirmation}
    return JsonResponse(data)


# credentials = service_account.Credentials.from_service_account_file(
#     "vision_credentials.json"
# )
# vision_client = vision.ImageAnnotatorClient(credentials=credentials)


# def find_vertices(byte_image, w, l, allow=True):

#     image = vision.Image(content=byte_image)
#     objects = vision_client.object_localization(
#         image=image
#     ).localized_object_annotations
#     shoes_coordinates = []
#     i = 1
#     for object_ in objects:
#         print(object_.name, object_.score)
#         if object_.name == "Shoe" or allow:
#             vertices = object_.bounding_poly.normalized_vertices
#             top_left = (int(vertices[0].x * w), int(vertices[0].y * l))
#             top_right = (int(vertices[1].x * w), int(vertices[1].y * l))
#             bottom_right = (int(vertices[2].x * w), int(vertices[2].y * l))
#             bottom_left = (int(vertices[3].x * w), int(vertices[3].y * l))
#             shoes_coordinates.append(
#                 [top_left, top_right, bottom_right, bottom_left, object_.name]
#             )
#             print(123, object_.name)
#     return shoes_coordinates
import base64


@csrf_exempt
@api_view(["GET"])
def ethwall_response(request):
    time_stamp = request.query_params.get("time-stamp", None)
    if not time_stamp:
        return HttpResponse("url not correct")
    image = cv2.imread(
        "/home/Charles85/super-admin/shoefitr/test_images2/" + time_stamp + ".jpg"
    )
    if image is None:
        return HttpResponse("Someting went wrong, try again")
    PPI = detect_square(image)
    size_data, found_flag, message, img = detect_fun(image, PPI=PPI)
    img_b64 = to_base64(img)
    try:
        left_length = int(size_data["length_l"])
        right_length = int(size_data["length_r"])
    except Exception as e:
        print(4443, e)
        return HttpResponse("Feet not found, try again")
    print(4444, left_length, right_length)

    if found_flag and img_b64:
        return render(
            request,
            "shoefitr/ethwall_response.html",
            {
                "image": img_b64,
                "left_length": left_length,
                "right_length": right_length,
            },
        )
    return HttpResponse("Feet not found")


def scan_ar(request):
    authenticated = False
    # webshop_client_ids = [str(user) for user in User.objects.all()]
    try:
        encoded_data = request.GET["time_stamp"]

    except Exception as e:
        return HttpResponse(e.args[0] + ", URL parameters are not correct!")

    context = {
        "time_stamp": encoded_data,
    }
    return render(request, "ar.html", context)


def new_page(request):
    authenticated = False
    # webshop_client_ids = [str(user) for user in User.objects.all()]

    context = {
        "time_stamp": "encoded_data",
    }
    return render(request, "instructions.html", context)


@csrf_exempt
def test_image(request):
    if request.method == "POST":
        try:
            picture = request.POST.get("picture", None)
            length_ppi = request.POST.get("length_ppi", None)
            width_ppi = request.POST.get("width_ppi", None)

            try:
                base64_picture = picture.split("base64,")[1]
            except:
                base64_picture = picture
            picture_stream = base64.decodebytes(base64_picture.encode("utf-8"))
            img = cv2.imdecode(
                np.frombuffer(picture_stream, np.uint8), cv2.IMREAD_UNCHANGED
            )
            now = datetime.now()
            time_stamp = now.strftime("%Y_%m_%d-%H_%M_%S.%f")[:-3]

            cv2.imwrite(
                "/home/Charles85/super-admin/shoefitr/test_images2/"
                + time_stamp
                + ".jpg",
                img,
            )
            combined_data = {
                "time_stamp": time_stamp,
                "length_ppi": length_ppi,
                "width_ppi": width_ppi,
            }
            json_data = json.dumps(combined_data).encode("utf-8")
            encoded_data = str(base64.b64encode(json_data), "UTF-8")

            print(45634, json_data, encoded_data, type(encoded_data))
            return JsonResponse(
                {
                    "success": True,
                    "redirect_url": "https://api.shoefitr.io/scan-ar?time_stamp="
                    + encoded_data,
                    "message": "success!",
                }
            )
        except Exception as e:
            print(234, e)
            return JsonResponse(
                {
                    "success": False,
                    "message": "Image not saved, send 'POST' request as base64 string with field named as 'picture'",
                }
            )
    return JsonResponse(
        {
            "success": False,
            "message": "'GET' method not allowed, send 'POST' request as base64 string with field named as 'picture'",
        }
    )


import os

import git


@csrf_exempt
def update_server(request):
    if request.method == "POST":
        path = "/home/Charles85/api/shoefitr/scan"
        os.chdir(path)
        print(os.popen("git stash").read())
        print(os.popen("git stash clear").read())
        repo = git.Repo(path)
        origin = repo.remotes.origin
        origin.pull()
        return HttpResponse("Updated PythonAnywhere successfully")
    return HttpResponse("Get Request not accepted!", status=400)


class MatchUserIdShopOwner(APIView):
    def post(self, request, *args, **kwargs):
        userid = request.data.get('userid')
        shop_owner_username = request.data.get('shopid')
        if not userid or not shop_owner_username:
            return Response({'error': 'userid and shopOwner are required'}, status=status.HTTP_400_BAD_REQUEST)
        matching_shoes = Shoes.objects.filter(userid=userid, shop__shopOwner__username=shop_owner_username)
        if matching_shoes.exists():
            shoe = matching_shoes.first()
            reference = shoe.reference
            reference_data = {
                'size': reference.size,
                'selection': reference.selection,
                'region': reference.region,
                'created_on': reference.created_on,
                'updated_on': reference.updated_on
            } if reference else None
            
            print(reference, reference_data)
            
            return Response({
                'match': True,
                'username': shoe.shop.shopOwner.username if shoe.shop else "",
                'reference': reference_data
            }, status=status.HTTP_200_OK)
        return Response({'match': False, 'username': '', 'reference': None}, status=status.HTTP_404_NOT_FOUND)


class ShopIDListView(APIView):
    def get(self, request):
        usernames = User.objects.all().values_list("username", flat=True)
        return Response(usernames, status=status.HTTP_200_OK)
    
class ModelNamesListView(APIView):
    def get(self, request, shopid):
        model_names, error = get_model_names_from_file(shopid)
        if error:
            return Response({"error": error}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        if model_names is None:
            return Response({"error": "File not found"}, status=status.HTTP_404_NOT_FOUND)
        return Response({"model_names": model_names}, status=status.HTTP_200_OK)