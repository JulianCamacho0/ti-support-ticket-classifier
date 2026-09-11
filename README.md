# ti-support-ticket-classifier
Sistema de ML para clasificar automáticamente tickets de soporte de TI a partir de su descripción en texto.

## Objetivo

Clasificar tickets de soporte de TI (a partir del asunto y la descripción del ticket) en la cola/área
de soporte (`queue`) a la que deben ser enrutados, mediante un pipeline de Machine Learning simple y
reproducible, pensado con fines académicos de aprendizaje de MLOps.

## Estructura del proyecto

```
data/                             # dataset de entrada
	dataset-ti-tickets.csv
models/                           # modelo entrenado (se genera al ejecutar el pipeline)
	ticket_classifier.joblib
src/
	config.py                       # rutas, constantes e hiperparámetros del proyecto
	schema.py                       # contrato de datos (Pandera)
	data.py                        # lectura del dataset
	validation.py                  # validación del dataset contra el contrato
	preprocessing.py               # construcción de features/target y pipeline sklearn
	train.py                       # entrenamiento del modelo
	evaluate.py                    # cálculo de métricas de evaluación
	main.py                        # punto de entrada del pipeline
tests/                            # pruebas unitarias (pytest)
requirements.txt
```

## Instalación de dependencias

```powershell
pip install -r requirements.txt
```

## Ubicación esperada de los datos

El pipeline espera el archivo `data/dataset-ti-tickets.csv` (ruta configurable en `src/config.py`).

## Cómo ejecutar el pipeline

```powershell
python src/main.py
```

## Qué hace cada etapa

1. **Lectura de datos** (`data.py`): carga el CSV en un `DataFrame`.
2. **Validación** (`validation.py` + `schema.py`): valida el `DataFrame` contra el contrato de datos
   definido con Pandera (columnas, tipos, valores permitidos, obligatoriedad). Si no cumple, el
   pipeline se detiene mostrando el error de validación.
3. **Preparación de features/target** (`preprocessing.py`): combina `subject` y `body` en un único
   texto de entrada (`X`) y separa la variable objetivo (`y`) `queue`.
4. **Train/test split** (`main.py`): separa los datos usando `TEST_SIZE` y `RANDOM_STATE` de `config.py`.
5. **Entrenamiento** (`train.py`): entrena un pipeline de scikit-learn (`TfidfVectorizer` + `LogisticRegression`).
6. **Evaluación** (`evaluate.py`): calcula accuracy, precision, recall, F1-score y matriz de confusión.
7. **Persistencia** (`main.py`): guarda el pipeline entrenado (vectorizador + modelo) con `joblib` en `models/`.

## Modelo utilizado

`LogisticRegression` sobre representación TF-IDF del texto (asunto + descripción del ticket). Es un
modelo baseline simple, sin búsqueda de hiperparámetros.

## Métricas generadas

Accuracy, precision, recall y F1-score (promedio ponderado), además de la matriz de confusión.

## Dónde queda almacenado el modelo

`models/ticket_classifier.joblib` (ruta configurable en `src/config.py`).
