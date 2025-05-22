import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import librosa
import numpy as np

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



def plot_segmented_genres(GENRE_MAP, results, figsize=(14, 4)):
    """
    Dibuja una gráfica con barras horizontales por modelo y segmento (Segmento 1, 2...), coloreadas según el género predicho.
    """

    # Configuración
    models = ['SVM', 'KNN', 'Red Neuronal']
    genre_colors = {}
    model_ypos = {model: i for i, model in enumerate(models)}
    bar_height = 0.25

    # Mapeo de colores por género
    unique_genres = list(set(GENRE_MAP.values()))
    cmap = plt.cm.get_cmap('tab10', len(unique_genres))
    for i, genre in enumerate(unique_genres):
        genre_colors[genre] = cmap(i)

    fig, ax = plt.subplots(figsize=figsize)

    # Dibujar barras por modelo y segmento
    for seg_idx, result in enumerate(results):
        label = f"Segmento {seg_idx + 1}"

        for model in models:
            if model not in result:
                continue

            genre = result[model].get('genre')
            if isinstance(genre, (np.int64, int)):
                genre = GENRE_MAP.get(genre, str(genre))

            color = genre_colors.get(genre, "gray")
            y = model_ypos[model]
            ax.bar(
                x=seg_idx,  # Usamos el índice como posición en X
                height=bar_height,
                width=0.9,
                bottom=y - bar_height / 2,
                color=color,
                edgecolor='black'
            )

    # Ajustes de ejes
    ax.set_yticks(list(model_ypos.values()))
    ax.set_yticklabels(models)
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels([f"Segmento {i + 1}" for i in range(len(results))], rotation=45, ha='right')
    ax.set_title("Género Predicho por Modelo en Cada Segmento")
    ax.set_xlabel("Segmentos")
    ax.set_ylabel("Modelos")

    # Leyenda
    legend_patches = [mpatches.Patch(color=color, label=genre) for genre, color in genre_colors.items()]
    ax.legend(handles=legend_patches, title="Géneros", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()
