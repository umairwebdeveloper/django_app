@echo off

REM Navigate to the virtual environment's Scripts directory
cd ../..
cd shoefitr-nextjs/backend/django_app/venv/Scripts

REM Activate the virtual environment
call activate

REM Navigate to the foot_api_yolov8-main directory
cd ../../../../..
cd foot_api_yolov8-main/foot_api_yolov8-main

REM Run the Django development server
python manage.py runserver

REM To keep the command window open after execution
cmd /k
