from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

model = load_model("mlp/modelo_dieta_dinossauros.h5")
preprocessor = joblib.load("mlp/dino_preprocessor.joblib")
label_encoder = joblib.load("mlp/dino_label_encoder.joblib")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Receber dados JSON
    data = request.get_json(force=True)

    # Criar um DataFrame com os dados recebidos
    new_dino = pd.DataFrame([data], index=[0])

    # Prepocessar, predizer e pegar o resultado
    new_dino_processed = preprocessor.transform(new_dino)
    prediction = model.predict(new_dino_processed)
    predicted_class_idx = np.argmax(prediction, axis=1)[0]
    predicted_diet = label_encoder.inverse_transform([predicted_class_idx])[0]

    response = {
        'predicted_diet': predicted_diet,
        'confidence': prediction[0].tolist(),
        'classes': label_encoder.classes_.tolist()
    }

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=False, port=5000)