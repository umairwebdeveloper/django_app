from django.conf import settings
import boto3
from botocore.exceptions import NoCredentialsError
import pandas as pd
from shoefitr.models import data
from django.contrib.auth import get_user_model

User = get_user_model()


def generate_presigned_url(file_name, expiration=3600):
    s3_client = boto3.client(
        "s3",
        region_name=settings.AWS_S3_REGION_NAME,
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    )
    try:
        response = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": settings.AWS_STORAGE_BUCKET_NAME, "Key": file_name},
            ExpiresIn=expiration,
        )
    except NoCredentialsError:
        return None
    return response


def get_model_names_from_file(shopid):
    file = data.objects.filter(shop__shopOwner__username=str(shopid)).order_by("-id").first()
    print("File: ", file)
    if not file:
        return None, "File not found"

    df = None  # Initialize df to None

    try:
        if ".xls" in str(file.file):
            df = pd.read_excel(file.file, header=0)
            print("--------xls")
        elif ".csv" in str(file.file):
            try:
                print("2----------csv")
                df = pd.read_csv(file.file, delimiter=";", header=0)
            except Exception as e:
                print("3----------csv", e)
                try:
                    df = pd.read_csv(file.file, delimiter=",", header=0)
                except Exception as e:
                    print(23)
                    print(e)
                    return None, str(e)

        if df is not None:
            model_names = df["Name"].unique()
            return model_names, None
        else:
            return None, "Failed to read the file"
    except Exception as e:
        print(e)
        return None, str(e)