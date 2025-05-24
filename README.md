# Proyecto Final - Inteligencia Artificial
# Clasificador de canciones según género


## Features Utilizadas
| Columna                      | Descripción                        |
| ---------------------------- |--------------------------------------------- |
| `filename`               | Nombre del archivo de audio correspondiente a la canción.|
| `length`                 | Duración total del archivo de audio en segundos.|
| `chroma_stft_mean`       | Promedio de la Transformada de Fourier a corto plazo (STFT) aplicada al vector cromático, representa la energía distribuida entre los 12 semitonos.|
| `chroma_stft_var`        | Varianza del vector cromático (STFT), refleja cuánta variabilidad hay entre semitonos a lo largo del tiempo.|
| `rms_mean`               | Promedio de la energía de la señal de audio (Root Mean Square).   |
| `rms_var`                    | Varianza de la energía RMS, muestra fluctuaciones en volumen/intensidad.             |
| `spectral_centroid_mean`     | Promedio del centroide espectral, indica el centro de masa del espectro (brillo del sonido). |
| `spectral_centroid_var`      | Varianza del centroide espectral.|
| `spectral_bandwidth_mean`    | Ancho medio del espectro, relacionado con la dispersión de frecuencias.         |
| `spectral_bandwidth_var`     | Varianza del ancho espectral.|
| `rolloff_mean`               | Frecuencia debajo de la cual se encuentra el 85% de la energía del espectro, promedio.|
| `rolloff_var`                | Varianza del rolloff espectral.|
| `zero_crossing_rate_mean`    | Promedio de veces que la señal cruza por cero, asociado a sonidos percusivos o ruidosos.|
| `zero_crossing_rate_var`     | Varianza del zero-crossing rate.|
| `harmony_mean`               | Promedio de la armonicidad estimada.|
| `harmony_var`                | Varianza de la armonicidad.|
| `perceptr_mean`              | Promedio del componente percusivo estimado.|
| `perceptr_var`               | Varianza del componente percusivo.|
| `tempo`                      | Estimación del tempo (BPM) de la canción.|
| `mfcc1_mean` a `mfcc20_mean` | Promedios de los 20 coeficientes cepstrales en las frecuencias de Mel (MFCC), que representan la envolvente del espectro de audio y son útiles para capturar características tímbricas.|
| `mfcc1_var` a `mfcc20_var`   | Varianzas de los coeficientes MFCC correspondientes.|
| `label`                      | Género musical de la canción (categoría objetivo).|

## 
``` bash
.
├───app  -------------------------------->  Aplicación Frontend
├───data  ------------------------------->  Todos los datos utilizados en el proyecto
│   ├───mapping  ------------------------>  Mapeo de los géneros musicales                           
│   ├───music  -------------------------->  Archivos .wav organizado por géneros 
│   ├───processed
│   │   ├───extracted_data3.csv     ----->  12 Features más valiosas de extracted_data3.csv
│   │   ├───extracted_data30.csv    ----->  12 Features más valiosas de extracted_data30.csv
│   │   ├───preprocessed_data3.csv  ----->  Features Escaladas y Géneros Codificados de features_3_sec.csv
│   │   └───preprocessed_data30.csv ----->  Features Escaladas y Géneros Codificados de features_30_sec.csv
│   ├───raw
│   │   ├───extracted_data3.csv     ----->  Features "crudas" extraídas de extract-from-0.ipynb
│   │   ├───extracted_data30.csv    ----->  Features "crudas" extraídas de extract-from-0.ipynb
│   │   ├───preprocessed_data3.csv  ----->  Features originales de Kaggle
│   │   └───preprocessed_data30.csv ----->  Features originales de Kaggle
│   ├───samples      -------------------->  Música de Prueba
│   └───spectrograms -------------------->  Sectrogramas extraídos de extract-from-0.ipynb
├───notebooks
│   ├───data_exploration.ipynb ---------->  Exploración de Datos
│   ├───extract-from-0.ipynb.ipynb ------>  Extracción de features de los archivos .wav y creación de los espectrogramas
│   ├───model_training.ipynb   ---------->  Entrenamiento de los modelos de clasificación
│   └───preprocessing.ipynb    ---------->  Preprocesamiento de los datos (Creación archivos .CSV de preprocessed y extracted data)
└───src
    ├───models       -------------------->  Modelos de clasificación
    │   ├───new_models  ----------------->  Modelos entrenados con los datos de extract-from-0.ipynb
    │   │   ├───knn.pkl  ---------------->  K-Nearest-Neighbors
    │   │   ├───neural_netword.pkl  ----->  Red Neuronal
    │   │   └───svm.pkl  ---------------->  Support  
    │   ├───knn.pkl  -------------------->  K-Nearest-Neighbors
    │   ├───neural_netword.pkl  --------->  Red Neuronal
    │   └───svm.pkl  -------------------->  Support Vector Machine
    ├───musica prueba  ------------------>  Archivos .wav de Prueba para el programa de Python
    └───__pycache__
```