import joblib
import sys
import matplotlib.pyplot as plt
import librosa
import numpy as np

def load_models(MODELS_PATH):
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

  
def print_results(results, audio_name, GENRE_MAP):
    """Imprime los resultados de clasificación de forma ordenada."""
    if not results:
        return

    print("\n" + "=" * 50)
    print(f"Resultados de clasificación para: {audio_name}")
    print("=" * 50)

    # Mostrar los 3 géneros principales por modelo
    for model_name, data in results.items():
        print(f"\n{model_name}:")
        for i, (genre, conf) in enumerate(data['top3'], 1):
            # Usar GENRE_MAP para mostrar el nombre del género si es número
            if isinstance(genre, (int, np.integer)):
                genre_name = GENRE_MAP.get(genre, str(genre))
            else:
                genre_name = str(genre)
            print(f"  {i}. {genre_name} (Confianza: {conf:.2f}%)")

    # Encontrar el género con mayor confianza entre todos los modelos
    best_genre = None
    best_conf = -1
    for data in results.values():
        genre = data['genre']
        conf = data['confidence']
        # Convertir a nombre si es necesario
        if isinstance(genre, (int, np.integer)):
            genre_name = GENRE_MAP.get(genre, str(genre))
        else:
            genre_name = str(genre)
        if conf > best_conf:
            best_conf = conf
            best_genre = genre_name

    print(f"\n>>> El género más probable según los modelos es: **{best_genre}** (Confianza: {best_conf:.2f}%) <<<")

