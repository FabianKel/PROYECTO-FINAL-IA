import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import joblib
import warnings

warnings.filterwarnings('ignore')

# Definir rutas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_PATH = os.path.join(BASE_DIR, 'src', 'models')

# Asegurar que las carpetas existan
os.makedirs(MODELS_PATH, exist_ok=True)

def load_data(file_name='preprocessed_data.csv'):
    """
    Carga los datos preprocesados y los divide en conjuntos de entrenamiento y prueba
    
    Args:
        file_name: Nombre del archivo CSV con los datos
        
    Returns:
        X_train, X_test, y_train, y_test, label_encoder
    """
    print(f"Cargando datos desde {os.path.join(DATA_PATH, file_name)}...")
    
    # Verificar si el archivo existe
    file_path = os.path.join(DATA_PATH, file_name)
    if not os.path.exists(file_path):
        available_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.csv')]
        if available_files:
            file_name = available_files[0]
            print(f"Archivo especificado no encontrado. Usando {file_name} en su lugar.")
            file_path = os.path.join(DATA_PATH, file_name)
        else:
            raise FileNotFoundError(f"No se encontraron archivos CSV en {DATA_PATH}")
    
    # Cargar los datos
    df = pd.read_csv(file_path)
    
    # Verificar columnas
    if 'label' in df.columns:
        label_col = 'label'
    elif 'label_encoded' in df.columns:
        label_col = 'label_encoded'
    else:
        raise ValueError("No se encontró una columna de etiquetas ('label' o 'label_encoded') en el DataFrame")
    
    # Preparar features y target
    X = df.drop(columns=[col for col in ['label', 'label_encoded', 'filename'] if col in df.columns])
    
    # Asegurarse de que las características sean numéricas
    X = X.select_dtypes(include=['number'])
    
    # Procesar etiquetas
    if label_col == 'label':
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(df[label_col])
        # Guardar el encoder para uso futuro
        joblib.dump(label_encoder, os.path.join(MODELS_PATH, 'label_encoder.pkl'))
    else:  # label_encoded
        y = df[label_col].values
        try:
            # Intentar cargar un encoder existente
            label_encoder = joblib.load(os.path.join(MODELS_PATH, 'label_encoder.pkl'))
        except:
            print("No se encontró un label_encoder guardado. Las etiquetas ya vienen codificadas.")
            label_encoder = None
    
    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Forma de los datos de entrenamiento: {X_train.shape}")
    print(f"Forma de los datos de prueba: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, label_encoder

def evaluate_model(model, model_name, X_train, X_test, y_train, y_test):
    """
    Entrena y evalúa un modelo, guardándolo en disco
    
    Args:
        model: Modelo a entrenar
        model_name: Nombre del modelo
        X_train, X_test, y_train, y_test: Datos
    
    Returns:
        model, accuracy, confusion_matrix
    """
    print(f"\nEntrenando modelo {model_name}...")
    model.fit(X_train, y_train)
    
    # Predicciones
    y_pred = model.predict(X_test)
    
    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy del modelo {model_name}: {accuracy:.4f}")
    
    # Informe de clasificación
    print("\nInforme de clasificación:")
    print(classification_report(y_test, y_pred))
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    
    # Guardar el modelo
    joblib.dump(model, os.path.join(MODELS_PATH, f'{model_name}.pkl'))
    
    # Guardar resultados para análisis posterior
    results = {
        'model_name': model_name,
        'accuracy': accuracy,
        'confusion_matrix': cm
    }
    
    # Guardar resultados como un archivo joblib
    joblib.dump(results, os.path.join(MODELS_PATH, f'{model_name}_results.pkl'))
    
    return model, accuracy, cm

def train_svm(X_train, X_test, y_train, y_test):
    """Entrenar un modelo SVM con búsqueda de hiperparámetros"""
    print("\n" + "="*50)
    print("Entrenando modelo SVM...")
    print("="*50)
    
    # Definir la búsqueda de hiperparámetros
    param_grid_svm = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.01, 0.1],
        'kernel': ['rbf', 'linear']
    }
    
    # Crear el modelo base
    svm_model = SVC(probability=True, random_state=42)
    
    # Configurar la búsqueda en grid
    svm_grid = GridSearchCV(
        svm_model,
        param_grid_svm,
        cv=StratifiedKFold(n_splits=3),  # Reducido para mayor rapidez
        scoring='accuracy',
        verbose=1,
        n_jobs=-1
    )
    
    # Realizar la búsqueda
    svm_grid.fit(X_train, y_train)
    
    # Mostrar los mejores parámetros
    print(f"Mejores parámetros para SVM: {svm_grid.best_params_}")
    print(f"Mejor puntuación de validación cruzada: {svm_grid.best_score_:.4f}")
    
    # Evaluar el mejor modelo
    best_svm = svm_grid.best_estimator_
    return evaluate_model(best_svm, 'svm', X_train, X_test, y_train, y_test)

def train_knn(X_train, X_test, y_train, y_test):
    """Entrenar un modelo KNN con búsqueda de hiperparámetros"""
    print("\n" + "="*50)
    print("Entrenando modelo KNN...")
    print("="*50)
    
    # Definir la búsqueda de hiperparámetros
    param_grid_knn = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }
    
    # Crear el modelo base
    knn_model = KNeighborsClassifier()
    
    # Configurar la búsqueda en grid
    knn_grid = GridSearchCV(
        knn_model,
        param_grid_knn,
        cv=StratifiedKFold(n_splits=3),  # Reducido para mayor rapidez
        scoring='accuracy',
        verbose=1,
        n_jobs=-1
    )
    
    # Realizar la búsqueda
    knn_grid.fit(X_train, y_train)
    
    # Mostrar los mejores parámetros
    print(f"Mejores parámetros para KNN: {knn_grid.best_params_}")
    print(f"Mejor puntuación de validación cruzada: {knn_grid.best_score_:.4f}")
    
    # Evaluar el mejor modelo
    best_knn = knn_grid.best_estimator_
    return evaluate_model(best_knn, 'knn', X_train, X_test, y_train, y_test)

def train_neural_network(X_train, X_test, y_train, y_test):
    """Entrenar una Red Neuronal con búsqueda de hiperparámetros"""
    print("\n" + "="*50)
    print("Entrenando modelo de Red Neuronal...")
    print("="*50)
    
    # Definir la búsqueda de hiperparámetros (reducida para mayor rapidez)
    param_grid_nn = {
        'hidden_layer_sizes': [(100,), (100, 50), (50, 30)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam'],
        'alpha': [0.0001, 0.001],
        'learning_rate': ['constant', 'adaptive'],
    }
    
    # Crear el modelo base
    nn_model = MLPClassifier(max_iter=300, random_state=42)
    
    # Configurar la búsqueda en grid
    nn_grid = GridSearchCV(
        nn_model,
        param_grid_nn,
        cv=StratifiedKFold(n_splits=3),  # Reducido para mayor rapidez
        scoring='accuracy',
        verbose=1,
        n_jobs=-1
    )
    
    # Realizar la búsqueda
    nn_grid.fit(X_train, y_train)
    
    # Mostrar los mejores parámetros
    print(f"Mejores parámetros para la Red Neuronal: {nn_grid.best_params_}")
    print(f"Mejor puntuación de validación cruzada: {nn_grid.best_score_:.4f}")
    
    # Evaluar el mejor modelo
    best_nn = nn_grid.best_estimator_
    return evaluate_model(best_nn, 'neural_network', X_train, X_test, y_train, y_test)

def train_models(data_file='preprocessed_data.csv'):
    """
    Entrenar todos los modelos
    
    Args:
        data_file: Nombre del archivo CSV con los datos
    """
    # Cargar los datos
    X_train, X_test, y_train, y_test, label_encoder = load_data(data_file)
    
    # Entrenar cada modelo
    svm_results = train_svm(X_train, X_test, y_train, y_test)
    knn_results = train_knn(X_train, X_test, y_train, y_test)
    nn_results = train_neural_network(X_train, X_test, y_train, y_test)
    
    # Recopilar resultados
    results = [
        {'model_name': 'SVM', 'accuracy': svm_results[1], 'confusion_matrix': svm_results[2]},
        {'model_name': 'KNN', 'accuracy': knn_results[1], 'confusion_matrix': knn_results[2]},
        {'model_name': 'Neural Network', 'accuracy': nn_results[1], 'confusion_matrix': nn_results[2]}
    ]
    
    # Ordenar los modelos por precisión
    results.sort(key=lambda x: x['accuracy'], reverse=True)
    
    # Mostrar resultados ordenados
    print("\n" + "="*50)
    print("Comparación de modelos")
    print("="*50)
    
    for i, result in enumerate(results):
        print(f"{i+1}. {result['model_name']}: {result['accuracy']:.4f}")
    
    # Guardar los resultados comparativos
    joblib.dump(results, os.path.join(MODELS_PATH, 'comparison_results.pkl'))
    
    print("\nProceso de entrenamiento completado correctamente.")
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Entrenamiento de modelos para clasificación de géneros musicales')
    parser.add_argument('--data', type=str, default='preprocessed_data.csv',
                        help='Archivo CSV con los datos preprocesados')
     
    args = parser.parse_args()
    
    train_models(args.data)