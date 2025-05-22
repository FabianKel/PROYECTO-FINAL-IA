import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import librosa
import math
import json

from feature_extraction import extract_all_features, extract_important_features


def classify_song(GENRE_MAP, audio_path, svm_model, knn_model, nn_model, label_encoder,
                  offset=None, duration=None, segmentation=False):
    """
    Clasifica una canción entera o por segmentos. Retorna resultados por modelo.
    """
    print(f"\nProcesando archivo: {audio_path}")

    try:
        if segmentation:
            total_duration = librosa.get_duration(path=audio_path)
            segment_length = duration or 30.0
            segment_num = int(total_duration // segment_length)
            segment_results = []

            for i in range(segment_num):
                offset_i = i * segment_length
                if i == segment_num - 1:
                    offset_i = max(0, total_duration - segment_length)

                result = classify_song(GENRE_MAP, audio_path, svm_model, knn_model, nn_model,
                                       label_encoder, offset=offset_i, duration=segment_length, segmentation=False)
                if result:
                    for model_name in result:
                        result[model_name]['segment'] = (offset_i, offset_i + segment_length)
                    segment_results.append(result)

            return segment_results  # Lista de resultados por segmento

        # Proceso normal de una porción o canción completa
        features = extract_important_features(audio_path, offset=offset, duration=duration)
        if features is None or len(features) == 0:
            print("No se pudieron extraer características del audio.")
            return None

        features_df = pd.DataFrame([features])
        model_features = svm_model.feature_names_in_
        for col in model_features:
            if col not in features_df.columns:
                features_df[col] = 0
        features_df = features_df[model_features]

        results = {}
        for model_name, model in zip(['SVM', 'KNN', 'Red Neuronal'], [svm_model, knn_model, nn_model]):
            proba = model.predict_proba(features_df)[0]
            top3_idx = np.argsort(proba)[::-1][:3]
            try:
                top3_genres = label_encoder.inverse_transform(top3_idx)
            except:
                top3_genres = [GENRE_MAP.get(int(idx), str(idx)) for idx in top3_idx]
            top3_conf = proba[top3_idx] * 100
            results[model_name] = {
                'top3': list(zip(top3_genres, top3_conf)),
                'genre': top3_genres[0],
                'confidence': top3_conf[0]
            }

        return results

    except Exception as e:
        print(f"Error al clasificar la canción: {e}")
        return None
