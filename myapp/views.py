from urllib import request
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage
import glob
import os
from inference import detect_fun
import time


@csrf_exempt
def index(request):
    folder = 'static/input_img/'
    url = request.get_host()
    if request.method == "POST" and request.FILES['image']:
        file = request.FILES.get('image')
        input_img = glob.glob('static/input_img/*')
        for f in input_img:
            os.remove(f)
          
        input_img = glob.glob('static/out_put/*')
        for f in input_img:
            os.remove(f)
           
        location = FileSystemStorage(location=folder)
        fn = location.save(file.name, file)
        path = os.path.join('static/input_img/', fn)
        length, respos_flag, ret , image_path = detect_fun(path)
        url_path = f'{url}/{image_path}'
        if respos_flag == True:
            context = {
                "message": ret,
                "length_foot": length,
                "Image_url": url_path
            }
            return JsonResponse(context, status=200)
        else:
            context = {
                "message": ret,
            }
            return JsonResponse(context, status=400)
        
    return render(request, 'index.html')