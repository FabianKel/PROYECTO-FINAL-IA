import os
import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import librosa

# Importamos el módulo para extraer características de audio
from feature_extraction import extract_features

# Definimos las rutas
MODELS_PATH = Path('../src/models/')
DATA_PATH = Path('../data/processed/')

GENRE_MAP = {
    0: "blues",
    1: "classical",
    2: "country",
    3: "disco",
    4: "hiphop",
    5: "jazz",
    6: "metal",
    7: "pop",
    8: "reggae",
    9: "rock"
}

def load_models():
    """Carga los modelos entrenados desde el directorio de modelos."""
    try:
        svm_model = joblib.load(MODELS_PATH / 'svm.pkl')
        knn_model = joblib.load(MODELS_PATH / 'knn.pkl')
        nn_model = joblib.load(MODELS_PATH / 'neural_network.pkl')
        label_encoder = joblib.load(MODELS_PATH / 'label_encoder.pkl')
        
        print("Modelos cargados exitosamente.")
        print("Clases del LabelEncoder:", label_encoder.classes_, type(label_encoder.classes_[0])) 
        return svm_model, knn_model, nn_model, label_encoder
    except Exception as e:
        print(f"Error al cargar los modelos: {e}")
        sys.exit(1)

def classify_song(audio_path, svm_model, knn_model, nn_model, label_encoder):
    """
    Clasifica una canción utilizando los tres modelos y muestra los 3 géneros más probables.
    """
    print(f"\nProcesando archivo: {audio_path}")
    try:
        features = extract_features(audio_path)
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

        for model_name, model in zip(
            ['SVM', 'KNN', 'Red Neuronal'],
            [svm_model, knn_model, nn_model]
        ):
            proba = model.predict_proba(features_df)[0]
            top3_idx = np.argsort(proba)[::-1][:3]
            try:
                top3_genres = label_encoder.inverse_transform(top3_idx)
            except:
                top3_genres = [GENRE_MAP.get(idx, str(idx)) for idx in top3_idx]
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

def print_results(results, audio_name):
    """Imprime los resultados de clasificación de forma ordenada."""
    if not results:
        return

    print("\n" + "=" * 50)
    print(f"Resultados de clasificación para: {audio_name}")
    print("=" * 50)

    for model_name, data in results.items():
        print(f"\n{model_name}:")
        for i, (genre, conf) in enumerate(data['top3'], 1):
            # Usar GENRE_MAP para mostrar el nombre del género si es número
            if isinstance(genre, (int, np.integer)):
                genre_name = GENRE_MAP.get(genre, str(genre))
            else:
                genre_name = str(genre)
            print(f"  {i}. {genre_name} (Confianza: {conf:.2f}%)")

def menu():
    """Muestra el menú principal de la aplicación."""
    print("\n" + "=" * 50)
    print("CLASIFICADOR DE GÉNEROS MUSICALES")
    print("=" * 50)
    print("1. Clasificar una canción")
    print("2. Clasificar todas las canciones en una carpeta")
    print("3. Salir")
    
    while True:
        try:
            option = int(input("\nSeleccione una opción (1-3): "))
            if 1 <= option <= 3:
                return option
            else:
                print("Opción no válida. Intente de nuevo.")
        except ValueError:
            print("Por favor, ingrese un número entre 1 y 3.")

def input_audio_path():
    """Solicita al usuario la ruta de un archivo de audio."""
    while True:
        audio_path = input("\nIngrese la ruta al archivo WAV (o 'q' para volver): ")
        if audio_path.lower() == 'q':
            return None
        
        # Eliminar comillas si el usuario las incluyó
        audio_path = audio_path.strip('"\'')
        
        # Convertir ruta relativa a absoluta si es necesario
        audio_path = os.path.abspath(audio_path)
        
        if not os.path.exists(audio_path):
            print(f"El archivo {audio_path} no existe. Intente de nuevo.")
            continue
            
        if not audio_path.lower().endswith('.wav'):
            print("El archivo debe tener formato WAV. Intente de nuevo.")
            continue
        
        print(f"Ruta válida: {audio_path}")    
        return audio_path

def input_folder_path():
    """Solicita al usuario la ruta de una carpeta con archivos de audio."""
    while True:
        folder_path = input("\nIngrese la ruta a la carpeta con archivos WAV (o 'q' para volver): ")
        if folder_path.lower() == 'q':
            return None
        
        # Eliminar comillas si el usuario las incluyó
        folder_path = folder_path.strip('"\'')
        
        # Convertir ruta relativa a absoluta si es necesario
        folder_path = os.path.abspath(folder_path)
        
        if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
            print(f"La carpeta {folder_path} no existe. Intente de nuevo.")
            continue
            
        wav_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.wav')]
        if not wav_files:
            print("No se encontraron archivos WAV en la carpeta. Intente de nuevo.")
            continue
        
        print(f"Carpeta válida con {len(wav_files)} archivos WAV: {folder_path}")    
        return folder_path

def show_audio_histogram(audio_path):
    """Muestra el histograma de amplitud del archivo de audio."""
    try:
        y, sr = librosa.load(audio_path, sr=None)
        plt.figure(figsize=(8, 4))
        plt.hist(y, bins=100, color='skyblue', edgecolor='black')
        plt.title('Histograma de amplitud del audio')
        plt.xlabel('Amplitud')
        plt.ylabel('Frecuencia')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"No se pudo mostrar el histograma: {e}")

def main():
    """Función principal del programa."""
    # Cargar los modelos
    print("Inicializando clasificador de géneros musicales...")
    svm_model, knn_model, nn_model, label_encoder = load_models()
    
    while True:
        option = menu()
        
        if option == 1:  # Clasificar una canción
            audio_path = input_audio_path()
            if audio_path:
                results = classify_song(audio_path, svm_model, knn_model, nn_model, label_encoder)
                print_results(results, os.path.basename(audio_path))
                # Mostrar histograma tras la clasificación
                show_audio_histogram(audio_path)
                
        elif option == 2:  # Clasificar todas las canciones en una carpeta
            folder_path = input_folder_path()
            if folder_path:
                wav_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.wav')]
                print(f"\nSe encontraron {len(wav_files)} archivos WAV para clasificar.")
                
                for filename in wav_files:
                    audio_path = os.path.join(folder_path, filename)
                    results = classify_song(audio_path, svm_model, knn_model, nn_model, label_encoder)
                    print_results(results, filename)
                    
                print("\nClasificación de carpeta completada.")
                
        elif option == 3:  # Salir
            print("\n¡Gracias por usar el clasificador de géneros musicales!")
            break
            
if __name__ == "__main__":
    main()