from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import os
import librosa
import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
import uvicorn
from audio_processing import extract_features, generate_spectrogram
import logging
import joblib
import pandas as pd

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
MODEL_DIR = Path("../models")
MUSIC_DIR = Path("../musica")
OUTPUT_DIR = Path("static")
OUTPUT_DIR.mkdir(exist_ok=True)

# Load model names
model_names = [
    "knn_original_3_preprocessed.pkl",
    "svm_actual_3_preprocessed.pkl",
    "knn_actual_3_preprocessed.pkl"
]
GENRE_MAP = {
    "0": "blues",
    "1": "classical",
    "2": "country",
    "3": "disco",
    "4": "hiphop",
    "5": "jazz",
    "6": "metal",
    "7": "pop",
    "8": "reggae",
    "9": "rock"
}

MODEL_FILES = [MODEL_DIR / name for name in model_names]

@app.get("/default-files")
async def get_default_files():
    """Return list of default .wav files in musica directory."""
    try:
        files = [f.name for f in MUSIC_DIR.glob("*.wav") if f.is_file()]
        print(files)
        return {"files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Configurar logging básico
logging.basicConfig(level=logging.INFO)


@app.post("/predict")
async def predict(file: UploadFile = File(None), default_file: str = None):
    """Process .wav file in 3-second segments and return aggregated predictions and spectrogram."""
    try:
        logging.info("Request received")

        # Determinar archivo de entrada
        if file:
            logging.info(f"Received uploaded file: {file.filename}")
            content = await file.read()
            temp_path = OUTPUT_DIR / file.filename
            with open(temp_path, "wb") as f:
                f.write(content)
            audio_path = temp_path
        elif default_file:
            audio_path = MUSIC_DIR / default_file
            logging.info(f"Using default file: {audio_path}")
            if not audio_path.exists():
                logging.error(f"Default file not found: {audio_path}")
                raise HTTPException(status_code=404, detail="Default file not found")
        else:
            logging.warning("No file or default_file provided")
            raise HTTPException(status_code=400, detail="No file provided")

        # Extraer características
        df_features = extract_features(str(audio_path))

        logging.info(f"DataFrame shape antes: {df_features.shape}")


        # Generar espectrograma
        logging.info("Generando spectograma...")
        spectrogram_path = generate_spectrogram(str(audio_path), OUTPUT_DIR)
        logging.info(f"Generated spectrogram: {spectrogram_path}")

        predictions = {}
        for model_file in MODEL_FILES:
            logging.info(f"Loading model: {model_file}")
            with open(model_file, "rb") as f:
                model = joblib.load(f)
            
            logging.info("modelo cargado exitosamente")

            logging.info(f"DataFrame shape después: {df_features.shape}")

            segment_probs = []
            for _, row in df_features.iterrows():
                features_array = row.values.reshape(1, -1)
                probs = model.predict_proba(features_array)[0]
                segment_probs.append(probs)
            avg_probs = np.mean(segment_probs, axis=0)
            logging.info(f"Promedio de Probabilidades: {avg_probs}")

            genres = model.classes_ if hasattr(model, 'classes_') else [f"Genre{i}" for i in range(len(avg_probs))]
            logging.info(f"Géneros: {genres}")
            predictions[model_file.stem] = {
                int(genre): float(prob) for genre, prob in zip(genres, avg_probs)
            }



        spectrogram_url = f"/static/{spectrogram_path.name}"
        logging.info("Prediction complete")

        logging.info(f"num_segments: {int(len(df_features))}")
        print("predictions type: ",type(predictions))
        print("spectrogram_url type: ",type(spectrogram_url))
        print(f"num_segments type: {type(int(len(df_features)))}, valor: {int(len(df_features))}")

        return {
            "predictions": {
                model_name: {
                    GENRE_MAP[str(genre)]: float(prob)
                    for genre, prob in genre_probs.items()
                }
                for model_name, genre_probs in predictions.items()
            },
            "spectrogram": spectrogram_url,
            "num_segments": int(len(df_features)),
            "message: ": "Return correcto"
        }

    except Exception as e:
        logging.exception("Error during prediction")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)