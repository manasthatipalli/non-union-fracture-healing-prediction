```python
from pathlib import Path
import time

import cv2
import numpy as np
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.db.models import Count
from django.shortcuts import render
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model

from .models import Prediction

MODEL_PATH = Path(settings.BASE_DIR) / "model" / "bone_fracture_resnet50_model.keras"
IMG_SIZE = 224

model = None


def get_model():
    global model

    if model is None:
        model = load_model(MODEL_PATH)

    return model


@login_required
def dashboard_view(request):
    return render(request, "dashboard/dashboard.html")


@login_required
def predict_view(request):
    result = None
    confidence_score = None
    inference_time = None
    error = None

    if request.method == "POST":
        uploaded_file = request.FILES.get("image")

        if uploaded_file:
            try:
                file_bytes = np.frombuffer(
                    uploaded_file.read(),
                    dtype=np.uint8
                )

                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                if image is None:
                    error = "The uploaded file is not a valid image."
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
                    image = preprocess_input(image.astype(np.float32))
                    image = np.expand_dims(image, axis=0)

                    start_time = time.time()
                    prediction = get_model().predict(image, verbose=0)
                    inference_time = round(time.time() - start_time, 3)

                    confidence = float(prediction[0][0])

                    if confidence < 0.5:
                        predicted_class = "Fractured"
                        confidence = 1 - confidence
                    else:
                        predicted_class = "Not Fractured"

                    confidence_score = round(confidence * 100, 2)
                    result = predicted_class

                    uploaded_file.seek(0)

                    Prediction.objects.create(
                        user=request.user,
                        input_file=uploaded_file,
                        predicted_class=predicted_class,
                        confidence=confidence_score
                    )
            except Exception:
                error = "Prediction failed. Please try again or contact the administrator."

        else:
            error = "Please upload an image."

    return render(
        request,
        "dashboard/predict.html",
        {
            "result": result,
            "confidence": confidence_score,
            "inference_time": inference_time,
            "error": error,
        },
    )


@login_required
def history_view(request):
    qs = Prediction.objects.filter(user=request.user)

    class_counts = qs.values("predicted_class").annotate(
        count=Count("id")
    )

    labels = [item["predicted_class"] for item in class_counts]
    data = [item["count"] for item in class_counts]

    return render(
        request,
        "dashboard/history.html",
        {
            "labels": labels,
            "data": data,
        },
    )


@login_required
def profile_page(request):
    profile = request.user.profile

    return render(
        request,
        "dashboard/profile.html",
        {
            "profile": profile,
        },
    )


@login_required
def my_predictions(request):
    predictions = Prediction.objects.filter(user=request.user)

    return render(
        request,
        "dashboard/my_predictions.html",
        {
            "predictions": predictions,
        },
    )
```
