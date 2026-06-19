from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from .models import Prediction
from django.db.models import Count

import time
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

MODEL_PATH = "model/bone_fracture_resnet50_model.keras"

model = load_model(MODEL_PATH)

IMG_SIZE = 224

@login_required
def dashboard_view(request):
    return render(request, 'dashboard/dashboard.html')

@login_required
def predict_view(request):

    result = None
    confidence_score = None

    if request.method == "POST":

        uploaded_file = request.FILES.get("image")

        if uploaded_file:

            # Convert uploaded image to OpenCV format
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

            image = np.array(image)

            image = preprocess_input(image)

            image = np.expand_dims(image, axis=0)

            start_time = time.time()

            prediction = model.predict(image)

            end_time = time.time()

            inference_time = end_time - start_time

            confidence = float(prediction[0][0])

            if confidence < 0.5:
                predicted_class = "Fractured"
                confidence = 1 - confidence
            else:
                predicted_class = "Not Fractured"

            confidence_score = round(confidence * 100, 2)

            result = predicted_class

            # Save prediction in database
            Prediction.objects.create(
                user=request.user,
                input_file=uploaded_file,
                predicted_class=predicted_class,
                confidence=confidence_score
            )

    return render(request, "dashboard/predict.html", {
        "result": result,
        "confidence": confidence_score
    })

@login_required
def history_view(request):
    qs = Prediction.objects.filter(user=request.user)

    # Count per class
    class_counts = qs.values('predicted_class').annotate(count=Count('id'))

    labels = [item['predicted_class'] for item in class_counts]
    data = [item['count'] for item in class_counts]

    return render(request, 'dashboard/history.html', {'labels': labels,'data': data,})

@login_required
def profile_page(request):
    profile = request.user.profile
    return render(request, 'dashboard/profile.html', {'profile': profile})

@login_required
def my_predictions(request):
    predictions = Prediction.objects.filter(user=request.user)
    return render(request, 'dashboard/my_predictions.html', {'predictions': predictions})
