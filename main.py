from fastapi import FastAPI, File, UploadFile, HTTPException
import cv2
import io
import os
import numpy as np
from keras.preprocessing import image as i1
from keras import models

app = FastAPI()

# Charger le modèle
model = models.load_model("model.h5")

# Fonction de classification
def predict_label(img_array):
    resized = cv2.resize(img_array, (50, 50))
    i = i1.img_to_array(resized) / 255.0
    i = i.reshape(1, 50, 50, 3)
    result = model.predict(i)
    a = round(result[0, 0], 2) * 100
    b = round(result[0, 1], 2) * 100
    probability = [a, b]
    threshold = 10

    if a > threshold or b > threshold:
        ind = np.argmax(result)
        classes = ["Cellule Normal: Pas de Paludisme", "Cellule Infecté :Présence du Paludisme"]
        return classes[ind], probability[ind]
    else:
        return "Invalid Image", 0

# Endpoint pour la classification d'une image sans stockage
@app.post("/predict/")
async def classify_image(file: UploadFile = File(...)):
    try:
        # Lecture du fichier en tant qu'image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image format")

        # Classification
        label, probability = predict_label(img)
        return {"filename": file.filename, "label": label, "probability": probability}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Route d'accueil
@app.get("/")
def home():
    return {"message": "Bienvenue sur l'API de classification du paludisme"}
