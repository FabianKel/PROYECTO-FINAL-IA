import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import librosa
import math
import json

# Importamos el módulo para extraer características de audio
from classify import classify_song
from plots import show_audio_histogram, plot_segmented_genres
from utils import load_models, print_results

# Definimos las rutas
MODELS_PATH = Path('../src/models/')
DATA_PATH = Path('../data/processed/')
MUSIC_PATH = Path('./musica prueba/')

# Se extrae el mapeo de géneros
with open('../data/mapping/genre_mapping_30.json', 'r', encoding='utf-8') as f:
    loaded_map = json.load(f)
GENRE_MAP = {int(k): v for k, v in loaded_map.items()}



  
def menu():
    """Muestra el menú principal de la aplicación."""
    print("\n" + "=" * 50)
    print("CLASIFICADOR DE GÉNEROS MUSICALES")
    print("=" * 50)
    print("1. Clasificar una canción")
    print("2. Clasificar una canción por partes (inicio y final)")
    print("3. Clasificar todas las canciones en una carpeta")
    print("4. Salir")
    
    while True:
        try:
            option = int(input("\nSeleccione una opción (1-4): "))
            if 1 <= option <= 4:
                return option
            else:
                print("Opción no válida. Intente de nuevo.")
        except ValueError:
            print("Por favor, ingrese un número entre 1 y 4.")

def input_audio_path():
    """Solicita al usuario la ruta de un archivo de audio."""
    while True:
        print("\nIngrese el número de la canción que desea clasificar (o 'q' para salir): ")
        song_list = os.listdir(MUSIC_PATH)
        for song in song_list:
            print(f"{song_list.index(song)} : {song}")
        audio_option = int(input("\nIngrese el número de la canción que desea clasificar (o 'q' para salir): "))
        

        if str(audio_option).lower() == 'q':
            return None
        
        if audio_option is not None and str(audio_option).isdigit():
            if audio_option >=0 and audio_option < len(song_list):
                
                if not song_list[audio_option].lower().endswith('.wav'):
                    print("El archivo debe tener formato WAV. Intente de nuevo.")
                    continue
        
                print(f"Archivo válido: {song_list[audio_option]}")
                song_path = os.path.join(MUSIC_PATH, song_list[audio_option])
                return song_path
        return None

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


def main():
    """Función principal del programa."""
    # Cargar los modelos
    print("Inicializando clasificador de géneros musicales...")
    svm_model, knn_model, nn_model, label_encoder = load_models(MODELS_PATH)
    
    while True:
        option = menu()
        
        if option == 1:  # Clasificar una canción
            audio_path = input_audio_path()
            if audio_path:
                results = classify_song(GENRE_MAP,audio_path, svm_model, knn_model, nn_model, label_encoder)
                print_results(results, os.path.basename(audio_path), GENRE_MAP)
                # Mostrar histograma tras la clasificación
                show_audio_histogram(audio_path)

        elif option == 2: # Clasificar una canción por partes (inicio y final)
            audio_path = input_audio_path()
            if audio_path:
                results = classify_song(GENRE_MAP, audio_path, svm_model, knn_model, nn_model,
                                        label_encoder, duration=30.0, segmentation=True)
                if results:
                    for i, res in enumerate(results):
                        print(f"\nSegmento {i + 1}")
                        print_results(res, os.path.basename(audio_path), GENRE_MAP)
                    plot_segmented_genres(GENRE_MAP, results)        

        elif option == 3:  # Clasificar todas las canciones en una carpeta
            folder_path = input_folder_path()
            if folder_path:
                wav_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.wav')]
                print(f"\nSe encontraron {len(wav_files)} archivos WAV para clasificar.")
                
                for filename in wav_files:
                    audio_path = os.path.join(folder_path, filename)
                    results = classify_song(GENRE_MAP,audio_path, svm_model, knn_model, nn_model, label_encoder)
                    print_results(results, filename, GENRE_MAP)
                    
                print("\nClasificación de carpeta completada.")

        elif option == 4:  # Salir
            print("\n¡Gracias por usar el clasificador de géneros musicales!")
            break
            
if __name__ == "__main__":
    main()