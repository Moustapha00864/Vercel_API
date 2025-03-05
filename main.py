import numpy as np
import cv2
import io 
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request

# Charger le modèle avec gestion des erreurs
try:
    model = load_model("model.h5", compile=False)
    print("Modèle chargé avec succès !")
except Exception as e:
    print(f"Erreur lors du chargement du modèle : {e}")
    exit()

# Classes des prédictions
CLASS_NAMES = ['Bénin', 'Malin']

def predict_label(img_path):
    try:
        # Charger et traiter l'image
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Erreur lors du chargement de l'image. Assurez-vous que le chemin est correct.")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir en RGB
        img = cv2.resize(img, (50, 50))  # Redimensionner selon les attentes du modèle
        img_array = np.array(img, dtype=np.float32) / 255.0  # Normaliser
        img_array = np.expand_dims(img_array, axis=0)  # Ajouter une dimension batch
        
        # Faire la prédiction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        return CLASS_NAMES[predicted_class]
    except Exception as e:
        print(f"Erreur de prédiction : {e}")
        return "Erreur de prédiction"

# Initialisation de Flask
app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template("index.html", prediction="Aucun fichier sélectionné")
        
        file = request.files['file']
        if file.filename == '':
            return render_template("index.html", prediction="Aucun fichier sélectionné")
        
        file_path = "static/uploads/" + file.filename
        file.save(file_path)
        
        prediction = predict_label(file_path)
        return render_template("index.html", prediction=prediction, img_path=file_path)
    
    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True)
