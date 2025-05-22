import numpy as np
import librosa
import warnings
import pandas as pd
import os

# Ignorar advertencias específicas de librosa
warnings.filterwarnings('ignore', category=UserWarning)

def extract_important_features(file_path, offset=None, duration=None):
    # Cargar el archivo de audio
    y, sr = librosa.load(file_path, sr=None, offset=offset, duration=duration)
    S = librosa.stft(y)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_stft_mean = np.mean(chroma_stft)
    chroma_stft_var = np.var(chroma_stft)
    rms = librosa.feature.rms(y=y)
    rms_mean = np.mean(rms)
    rms_var = np.var(rms)
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_centroid_mean = np.mean(spec_centroid)
    spectral_centroid_var = np.var(spec_centroid)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_bandwidth_mean = np.mean(spec_bw)
    spectral_bandwidth_var = np.var(spec_bw)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rolloff_mean = np.mean(rolloff)
    rolloff_var = np.var(rolloff)
    zcr = librosa.feature.zero_crossing_rate(y)
    zero_crossing_rate_mean = np.mean(zcr)
    zero_crossing_rate_var = np.var(zcr)
    harmony = librosa.effects.harmonic(y)
    harmony_mean = np.mean(harmony)
    harmony_var = np.var(harmony)
    perceptr = librosa.feature.melspectrogram(y=y, sr=sr)
    perceptr_mean = np.mean(librosa.power_to_db(perceptr, ref=np.max))
    perceptr_var = np.var(librosa.power_to_db(perceptr, ref=np.max))
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_means = [np.mean(m) for m in mfccs]
    mfcc_vars = [np.var(m) for m in mfccs]


    features = {
        'song_title': os.path.basename(file_path),
        'chroma_stft_mean': chroma_stft_mean,
        'chroma_stft_var': chroma_stft_var,
        'rms_mean': rms_mean,
        'rms_var': rms_var,
        'spectral_centroid_mean': spectral_centroid_mean,
        'spectral_centroid_var': spectral_centroid_var,
        'spectral_bandwidth_mean': spectral_bandwidth_mean,
        'spectral_bandwidth_var': spectral_bandwidth_var,
        'rolloff_mean': rolloff_mean,
        'rolloff_var': rolloff_var,
        'zero_crossing_rate_mean': zero_crossing_rate_mean,
        'zero_crossing_rate_var': zero_crossing_rate_var,
        'harmony_mean': harmony_mean,
        'harmony_var': harmony_var,
        'perceptr_mean': perceptr_mean,
        'perceptr_var': perceptr_var,
        'tempo': tempo,
    }
    # Agregar MFCCs
    for i in range(20):
        features[f'mfcc{i+1}_mean'] = mfcc_means[i]
        features[f'mfcc{i+1}_var'] = mfcc_vars[i]

    # Guardar las features para comprobar los valores
    # df = pd.DataFrame([features])
    # df.to_csv('features.csv', index=False)

    return features


def extract_all_features(file_path, n_mfcc=20, n_chroma=12, n_spectral_contrast=7, offset=None, duration=None):
    """
    Extrae características de audio de un archivo WAV utilizando librosa.
    Esta función debe ser idéntica o compatible con la utilizada para entrenar los modelos.
    
    Args:
        file_path: Ruta al archivo de audio WAV
        n_mfcc: Número de coeficientes MFCC a extraer
        n_chroma: Número de características de croma
        n_spectral_contrast: Número de bandas para el contraste espectral
        
    Returns:
        Diccionario con las características extraídas
    """
    try:
        # Cargar archivo de audio y normalizar
        print(f"Cargando archivo de audio: {file_path}")
        y, sr = librosa.load(file_path, sr=None, offset=offset, duration=duration)
        
        # Asegurar una duración mínima del audio para extraer características
        if len(y) < sr * 0.1:  # Si es menor a 0.1 segundos
            print("El archivo de audio es demasiado corto para extraer características.")
            return None
        
        print(f"Audio cargado: {len(y)/sr:.2f} segundos, frecuencia: {sr} Hz")
        
        # Definir diccionario para almacenar las características
        features = {}
        
        # Extraer características temporales
        print("Extrayendo características temporales...")
        
        # Valor RMS (energía)
        features['rms_mean'] = np.mean(librosa.feature.rms(y=y)[0])
        features['rms_var'] = np.var(librosa.feature.rms(y=y)[0])
        
        # Tasa de cruce por cero
        features['zero_crossing_rate_mean'] = np.mean(librosa.feature.zero_crossing_rate(y=y)[0])
        features['zero_crossing_rate_var'] = np.var(librosa.feature.zero_crossing_rate(y=y)[0])
        
        # Extraer características espectrales
        print("Extrayendo características espectrales...")
        
        # MFCCs (Mel-frequency cepstral coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        for i in range(n_mfcc):
            features[f'mfcc{i+1}_mean'] = np.mean(mfccs[i])
            features[f'mfcc{i+1}_var'] = np.var(mfccs[i])
        
        # Centroide espectral
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_var'] = np.var(spectral_centroids)
        
        # Rolloff espectral
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_rolloff_var'] = np.var(spectral_rolloff)
        
        # Ancho de banda espectral
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        features['spectral_bandwidth_var'] = np.var(spectral_bandwidth)
        
        # Características de croma
        chromagram = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=n_chroma)
        for i in range(n_chroma):
            features[f'chroma{i+1}_mean'] = np.mean(chromagram[i])
            features[f'chroma{i+1}_var'] = np.var(chromagram[i])
        
        # Contraste espectral
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=n_spectral_contrast)
        for i in range(n_spectral_contrast):
            features[f'spectral_contrast{i+1}_mean'] = np.mean(spectral_contrast[i])
            features[f'spectral_contrast{i+1}_var'] = np.var(spectral_contrast[i])
        
        # Extraer características de tiempo
        print("Extrayendo características rítmicas...")
        
        # Tempo
        try:
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            features['tempo'] = tempo
        except:
            features['tempo'] = 0
        
        print(f"Extracción completada: {len(features)} características extraídas.")
        return features
    
    except Exception as e:
        print(f"Error al extraer características: {e}")
        return None

def extract_features_to_dataframe(file_path):
    """
    Extrae características de audio y las devuelve como DataFrame.
    
    Args:
        file_path: Ruta al archivo de audio WAV
        
    Returns:
        DataFrame con las características extraídas
    """
    features = extract_all_features(file_path)
    if features:
        return pd.DataFrame([features])
    return None

if __name__ == "__main__":
    # Ejemplo de uso
    import sys
    
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        print(f"Extrayendo características de {audio_file}...")
        features = extract_all_features(audio_file)
        
        if features:
            print("\nCaracterísticas extraídas:")
            for k, v in features.items():
                print(f"{k}: {v}")
    else:
        print("Uso: python feature_extraction.py <ruta_archivo_wav>")