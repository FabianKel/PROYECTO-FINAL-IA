import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from tkinter import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

def plot_audio_features(file_path):

    # Cargar archivo de audio
    y, sr = librosa.load(file_path, sr=None)
    
    # Crear ventana Tkinter
    root = Tk()
    root.title(f"Análisis de Audio: {file_path}")
    root.geometry("1200x800")
    root.configure(background='white')
    
    # Frame principal
    main_frame = Frame(root, bg='white')
    main_frame.pack(fill="both", expand=True, padx=10, pady=10)
    
    # Crear figura y ejes
    fig = Figure(figsize=(12, 8), dpi=100)
    
    # Añadir subplots
    ax1 = fig.add_subplot(311)  # Forma de onda
    ax2 = fig.add_subplot(312)  # Espectrograma
    ax3 = fig.add_subplot(313)  # Cromograma
    
    # Configurar el espacio entre subplots
    fig.subplots_adjust(hspace=0.4)
    
    # Visualización 1: Forma de onda
    librosa.display.waveshow(y, sr=sr, ax=ax1)
    ax1.set_title('Forma de Onda')
    ax1.set_xlabel('Tiempo (s)')
    ax1.set_ylabel('Amplitud')
    
    # Visualización 2: Espectrograma
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sr, ax=ax2)
    ax2.set_title('Espectrograma')
    ax2.set_xlabel('Tiempo (s)')
    ax2.set_ylabel('Frecuencia (Hz)')
    fig.colorbar(img, ax=ax2, format='%+2.0f dB')
    
    # Visualización 3: Cromograma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    img2 = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', sr=sr, ax=ax3)
    ax3.set_title('Cromograma')
    ax3.set_xlabel('Tiempo (s)')
    ax3.set_ylabel('Pitch Class')
    fig.colorbar(img2, ax=ax3)
    
    # Crear canvas para mostrar la figura en Tkinter
    canvas = FigureCanvasTkAgg(fig, master=main_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=BOTH, expand=True)
    
    # Información adicional del audio
    info_frame = Frame(main_frame, bg='white')
    info_frame.pack(fill="x", pady=5)

    # Añadir etiquetas de información
    duration = len(y) / sr
    Label(info_frame, text=f"Duración: {duration:.2f} segundos", bg='white', font=("Arial", 12)).pack(side="left", padx=10)
    Label(info_frame, text=f"Frecuencia de muestreo: {sr} Hz", bg='white', font=("Arial", 12)).pack(side="left", padx=10)
    
    # Botón para cerrar la ventana
    Button(main_frame, text="Cerrar", command=root.destroy, font=("Arial", 12), bg="#ff5722", fg="white").pack(pady=10)
    
    # Iniciar el bucle principal de Tkinter
    root.mainloop()

def visualize_waveform_and_features(file_path):
    # Cargar archivo de audio
    y, sr = librosa.load(file_path, sr=None)
    
    # Crear figura y ejes
    plt.figure(figsize=(15, 10))
    
    # Visualización 1: Forma de onda
    plt.subplot(3, 1, 1)
    librosa.display.waveshow(y, sr=sr)
    plt.title('Forma de Onda')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud')
    
    # Visualización 2: Espectrograma
    plt.subplot(3, 1, 2)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sr)
    plt.title('Espectrograma')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Frecuencia (Hz)')
    plt.colorbar(format='%+2.0f dB')
    
    # Visualización 3: Cromograma
    plt.subplot(3, 1, 3)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', sr=sr)
    plt.title('Cromograma')
    plt.xlabel('Tiempo (s)')
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        print(f"Visualizando {audio_file}...")
        plot_audio_features(audio_file)
    else:
        print("Uso: python audio_visualization.py <ruta_archivo_wav>")