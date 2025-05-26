from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles  
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

app.mount("/static", StaticFiles(directory=str(OUTPUT_DIR)), name="static")
app.mount("/musica", StaticFiles(directory=str(MUSIC_DIR)), name="musica")

# Load model names
model_names = [
    "knn_original_3_preprocessed.pkl",
    "svm_original_3_preprocessed.pkl",
    "nn_original_3_preprocessed.pkl"
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
        logging.info(f"Found default files: {files}")
        logging.info(f"Music directory path: {MUSIC_DIR.absolute()}")
        logging.info(f"Music directory exists: {MUSIC_DIR.exists()}")
        return {"files": files}
    except Exception as e:
        logging.error(f"Error getting default files: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Configurar logging básico
logging.basicConfig(level=logging.INFO)

@app.post("/predict")
async def predict(
    file: UploadFile = File(None), 
    default_file: str = Form(None)
):
    """Process .wav file in 3-second segments and return aggregated predictions and spectrogram."""
    try:
        logging.info("Request received")
        logging.info(f"File: {file.filename if file else 'None'}")
        logging.info(f"Default file: {default_file}")

        # Determinar archivo de entrada
        if file and file.filename:
            logging.info(f"Received uploaded file: {file.filename}")
            content = await file.read()
            temp_path = OUTPUT_DIR / file.filename
            with open(temp_path, "wb") as f:
                f.write(content)
            audio_path = temp_path
            
        elif default_file:
            audio_path = MUSIC_DIR / default_file
            logging.info(f"Using default file: {audio_path}")
            logging.info(f"Default file exists: {audio_path.exists()}")
            if not audio_path.exists():
                logging.error(f"Default file not found: {audio_path}")
                raise HTTPException(status_code=404, detail=f"Default file not found: {default_file}")
                
        else:
            logging.warning("No file or default_file provided")
            raise HTTPException(status_code=400, detail="No file provided")

        logging.info(f"Processing audio file: {audio_path}")

        # Extraer características
        df_features = extract_features(str(audio_path))
        logging.info(f"DataFrame shape: {df_features.shape}")

        # Generar espectrograma
        logging.info("Generando espectrograma...")
        spectrogram_path = generate_spectrogram(str(audio_path), OUTPUT_DIR)
        logging.info(f"Generated spectrogram: {spectrogram_path}")

        if not spectrogram_path.exists():
            logging.error(f"Spectrogram not created: {spectrogram_path}")
            raise HTTPException(status_code=500, detail="Failed to generate spectrogram")

        predictions = {}
        for model_file in MODEL_FILES:
            logging.info(f"Loading model: {model_file}")
            try:
                with open(model_file, "rb") as f:
                    model = joblib.load(f)
                
                logging.info("Modelo cargado exitosamente")

                segment_probs = []
                for _, row in df_features.iterrows():
                    features_array = row.values.reshape(1, -1)
                    probs = model.predict_proba(features_array)[0]
                    segment_probs.append(probs)
                    
                avg_probs = np.mean(segment_probs, axis=0)
                logging.info(f"Promedio de Probabilidades: {avg_probs}")

                genres = model.classes_ if hasattr(model, 'classes_') else [f"Genre{i}" for i in range(len(avg_probs))]
                logging.info(f"Géneros: {genres}")

                # Si las clases son números, usa GENRE_MAP; si son strings, úsalas directamente
                if all(str(g).isdigit() for g in genres):
                    genre_names = [GENRE_MAP[str(g)] for g in genres]
                else:
                    genre_names = [str(g) for g in genres]

                predictions[model_file.stem] = {
                    genre: float(prob) for genre, prob in zip(genre_names, avg_probs)
                }

            except Exception as model_error:
                logging.error(f"Error loading model {model_file}: {model_error}")
                continue

        if not predictions:
            raise HTTPException(status_code=500, detail="No models could be loaded successfully")

        # CONSTRUIR URL CORRECTA DEL ESPECTROGRAMA
        spectrogram_url = f"http://localhost:8000/static/{spectrogram_path.name}"
        logging.info(f"Spectrogram URL: {spectrogram_url}")

        response_data = {
            "predictions": predictions,
            "spectrogram": spectrogram_url,
            "num_segments": int(len(df_features)),
            "message": "Predicción completada exitosamente"
        }
        
        logging.info(f"Returning response with spectrogram: {spectrogram_url}")
        return response_data

    except Exception as e:
        logging.exception("Error during prediction")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/check-music-files")
async def check_music_files():
    """Endpoint para debugging - verificar archivos de música disponibles."""
    try:
        music_files = []
        if MUSIC_DIR.exists():
            for file_path in MUSIC_DIR.glob("*.wav"):
                music_files.append({
                    "name": file_path.name,
                    "path": str(file_path),
                    "exists": file_path.exists(),
                    "size": file_path.stat().st_size if file_path.exists() else 0
                })
        
        return {
            "music_directory": str(MUSIC_DIR.absolute()),
            "directory_exists": MUSIC_DIR.exists(),
            "files": music_files
        }
    except Exception as e:
        logging.error(f"Error checking music files: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)