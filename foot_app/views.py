import cv2
import numpy as np
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from utils.specific import (
    calculate_model_id,
    image_resize,
    save_to_db,
    to_base64,
    user_inputs,
)

from .snippets import detect_fun

ACCEPTABLE_FEET_DIFFERENCE = 17  # in mm


@csrf_exempt
def calculation(request):
    if request.method == "POST":
        shopid = request.POST.get("shopid", None)
        userid = request.POST.get("userid", None)
        # marginid = request.POST.get("marginid", None)
        model_name = request.POST.get("model_name", None)
        size = round(float(request.POST.get("size", 0)), 2)
        selection = request.POST.get("selection", None)
        region = request.POST.get("system", None)
        picture_file = request.FILES.get("picture", None)
        img = cv2.imdecode(
            np.frombuffer(picture_file.read(), np.uint8), cv2.IMREAD_UNCHANGED
        )

        # now = datetime.now()
        # time = now.strftime("%d_%m_%Y %H_%M_%S")
        # try:
        #     cv2.imwrite(
        #         "/home/Charles85/super-admin/shoefitr/test_images/new/"
        #         + size
        #         + "_"
        #         + system
        #         + "_"
        #         + time
        #         + ".jpg",
        #         img,
        #     )
        # except Exception as e:
        #     print(234, e)
        # l, w, _ = img.shape

        len_size = user_inputs(size, region, selection)
        size_data, found, message, arrowed_image = detect_fun(img, len_size)

        # message, size_data, img_data, found = magic(img, size, system, adult)
        print(8889, message, size_data, found)

        # find_coordinates = False
        # if find_coordinates:
        #     coordinates = None  # find_vertices(byte_image, w, l)

        #     for c in coordinates:
        #         cv2.rectangle(img, c[0], c[2], color=(0, 255, 0), thickness=5)
        #         cv2.putText(
        #             img,
        #             c[4],
        #             (c[0][0] + 5, c[0][1] + 20),
        #             cv2.FONT_HERSHEY_COMPLEX_SMALL,
        #             1.5,
        #             (0, 0, 250),
        #             2,
        #         )
        # found = False
        correct_size = "_"
        try:
            if (
                abs(int(size_data["length_l"]) - int(size_data["length_r"]))
                > ACCEPTABLE_FEET_DIFFERENCE
            ):
                found = False
                message = "Feet not detected properly, Place both feet on the ground and scan again!"
                print(32423, message)
        except Exception as e:
            print(23433, e)
            found = False
            message = "Feet not detected properly, Place both feet on the ground and scan again!"

        if found:
            if arrowed_image is None:
                base64_string = to_base64(img)
            elif arrowed_image is not None:
                resized_arrowed_image = image_resize(arrowed_image, width=300)
            base64_string = to_base64(resized_arrowed_image)
            if int(size_data["length_l"]) > int(size_data["length_r"]):
                length = int(size_data["length_l"])
                width = int(size_data["waist_l"])
                ball = int(size_data["ball_l"])
                instep = int(size_data["instep_l"])
            else:
                length = int(size_data["length_r"])
                width = int(size_data["waist_r"])
                ball = int(size_data["ball_r"])
                instep = int(size_data["instep_r"])
            data = calculate_model_id(
                shopid, model_name, length, width, ball, instep, marginid=None
            )
            print(32342, data)
            if length < 1 or width < 1:
                size = (
                    width_advice
                ) = ball_advice = instep_advice = picture_advice = modelid = "_"
                print(3234, "length not found")
            else:

                def get_advice_in_color(advice):
                    if advice == "Tight":
                        return ("r", False)
                    if advice == "Loose":
                        return ("g", False)
                    return ("w", True)

                correct_size = data["size"]
                width_advice = data["width_advice"]
                ball_advice = data["ball_advice"]
                instep_advice = data["instep_advice"]
                width_color, width_fit = get_advice_in_color(width_advice)
                ball_color, ball_fit = get_advice_in_color(ball_advice)
                instep_color, instep_fit = get_advice_in_color(instep_advice)
                totally_fit = width_fit and ball_fit and instep_fit
                picture_advice = f"i-{instep_color}_w-{width_color}_b-{ball_color}"
                modelid = data["model_id"]
                save_to_db(
                    shopid,
                    userid,
                    modelid,
                    size_data["length_l"],
                    size_data["length_r"],
                    size_data["waist_l"],
                    size_data["waist_r"],
                    size_data["instep_l"],
                    size_data["instep_r"],
                    size_data["ball_l"],
                    size_data["ball_r"],
                    correct_size,
                    width_advice,
                    ball_advice,
                    instep_advice,
                    model_name,
                    size,
                    selection,
                    region,
                    arrowed_image,
                    None,
                )
            response_data = {
                "uri": base64_string,
                "found": found,
                "width_advice": width_advice,
                "ball_advice": ball_advice,
                "instep_advice": instep_advice,
                "picture_advice": picture_advice,
                "width_fit": width_fit,
                "ball_fit": ball_fit,
                "instep_fit": instep_fit,
                "totally_fit": totally_fit,
                "model_id": modelid,
                "size": size,
                "correct_size": correct_size,
                "message": message,
            }
            response_data.update(size_data)
            print(22223, "saved to db")
            return JsonResponse(response_data)
        else:
            length = None
            width = None
            width_advice = "_"
            modelid = "_"
            response_data = {
                "uri": "",
                "found": found,
                "width_advice": width_advice,
                "model_id": modelid,
                "size": size,
                "correct_size": correct_size,
                "message": message,
            }
            print(32323, "Errrorr", message)

            return JsonResponse(response_data)

