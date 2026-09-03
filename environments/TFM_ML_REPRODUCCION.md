# Reproducción del entorno TFM_ML

## 1. Finalidad

`TFM_ML` es el entorno principal del Trabajo Fin de Máster sobre
detección preventiva de indicadores de riesgo en salud mental adulta
mediante aprendizaje automático multisalida a partir de NSDUH 2024.

Se utilizará para preparación de datos, Machine Learning, tratamiento
del desbalanceo, XGBoost, SHAP y Deep Learning tabular con TensorFlow
y Keras.

Los agentes de inteligencia artificial generativa se configurarán
posteriormente en un entorno independiente.

## 2. Ubicaciones

Proyecto técnico en WSL:

    /home/cam/BD/TFM_NSDUH

Documentación en Windows:

    C:\Users\CAM\Documents\TFM_NSDUH

Entorno Conda:

    /home/cam/miniconda3/envs/TFM_ML

Kernel de Jupyter:

    Python (TFM_ML)

JupyterLab se ejecuta desde el entorno `base`; los notebooks del TFM
usan el kernel `Python (TFM_ML)`.

## 3. Versiones principales validadas

- Python 3.11.15
- NumPy 2.4.6
- pandas 3.0.5
- PyArrow 25.0.0
- openpyxl 3.1.5
- pypdf 6.14.2
- pydot 4.0.1
- SciPy 1.17.1
- scikit-learn 1.9.0
- imbalanced-learn 0.14.2
- XGBoost 3.2.0
- SHAP 0.51.0
- Matplotlib 3.11.1
- Seaborn 0.13.2
- Joblib 1.5.3
- TensorFlow 2.20.0
- Keras 3.15.1
- TensorBoard 2.20.0
- ipykernel 7.3.0
- ipywidgets 8.1.8

## 4. GPU validada

- NVIDIA RTX 1000 Ada Generation Laptop GPU
- Capacidad de cómputo 8.9
- TensorFlow compilado para CUDA 12.5.1
- cuDNN 9
- Operaciones y predicciones ejecutadas en `/GPU:0`

TensorFlow 2.21.0 fue descartado porque no registraba la GPU en este
equipo. TensorFlow 2.20.0 fue validado mediante cálculo matricial,
entrenamiento multisalida, predicción y restauración de modelos.

## 5. Reconstrucción del entorno

Desde la raíz del proyecto:

    cd "$HOME/BD/TFM_NSDUH"

Crear el entorno:

    conda env create \
        --file environments/TFM_ML_environment.yml

## 6. Configuración de CUDA

Después de crear el entorno o reinstalar TensorFlow:

    bash environments/setup_tensorflow_cuda_links.sh TFM_ML

El script reconstruye los enlaces simbólicos de las bibliotecas NVIDIA
y puede ejecutarse repetidamente.

## 7. Registro del kernel

    conda run --no-capture-output -n TFM_ML \
        python -m ipykernel install \
        --user \
        --name tfm_ml \
        --display-name "Python (TFM_ML)"

Comprobación:

    conda run --no-capture-output -n base \
        jupyter kernelspec list

## 8. Prueba integral

Desde la raíz del proyecto:

    conda run --no-capture-output -n TFM_ML \
        python tests/smoke_test_environment.py

La prueba valida versiones, GPU, pila científica, widgets, Machine
Learning multisalida, desbalanceo, XGBoost, SHAP, Deep Learning
multisalida y persistencia de modelos.

## 9. Apertura de JupyterLab

    bdlab "$HOME/BD/TFM_NSDUH"

Seleccionar el kernel:

    Python (TFM_ML)

La raíz debe ser:

    /home/cam/BD/TFM_NSDUH

## 10. Archivos de reproducibilidad

Se conservan únicamente los elementos necesarios:

- `TFM_ML_environment.yml`:
  archivo de configuración con las dependencias directas del entorno.

- `TFM_ML_environment_lock.yml`:
  congelación completa del entorno Conda y de las dependencias pip,
  incluyendo las versiones y compilaciones utilizadas.

- `TFM_ML_requirements_lock.txt`:
  congelación complementaria de las distribuciones Python gestionadas
  mediante pip.

- `setup_tensorflow_cuda_links.sh`:
  reconstrucción de los enlaces CUDA requeridos por TensorFlow.

- `../tests/smoke_test_environment.py`:
  prueba integral de versiones, Machine Learning, Deep Learning y GPU.

El entorno `TFM_ML` quedó definitivamente cerrado y congelado el
26/08/2026, una vez finalizadas las fases de Machine Learning, Deep
Learning, evaluación final y explicabilidad mediante SHAP.

`TFM_ML_environment.yml` conserva las dependencias directas necesarias
para reconstruir el entorno de forma legible. La reproducción del estado
completo validado se apoya en los archivos de congelación indicados a
continuación.

Las versiones principales validadas permanecen documentadas en este
archivo y también se comprueban automáticamente mediante el test
integral.

Graphviz disponible en WSL no se declara en `TFM_ML_environment.yml`,
porque es una dependencia del sistema. Para la representación de modelos,
se utiliza `pydot` en el entorno y Graphviz a nivel de sistema, con el
ejecutable `dot` validado en `/usr/bin/dot`.


## 11. Comprobaciones rápidas

Dependencias:

    conda run --no-capture-output -n TFM_ML \
        python -m pip check

GPU del sistema:

    nvidia-smi

GPU desde TensorFlow:

    conda run --no-capture-output -n TFM_ML \
        python -c \
        'import tensorflow as tf; print(tf.config.list_physical_devices("GPU"))'

## 12. Reglas de trabajo

- Datos, notebooks y resultados técnicos se almacenan en WSL.
- No se trabaja directamente desde `/mnt/c` con datos o notebooks.
- Los documentos DOCX se almacenan en Windows.
- Los archivos originales permanecen inmutables en `data/raw`.
- La programación se desarrolla y valida paso a paso.
- No se crean carpetas permanentes hasta que tengan contenido real.

## 13. Estado final del entorno

Estado: **CERRADO Y CONGELADO**

Fecha de congelación: **26/08/2026**

El entorno fue validado antes de su congelación mediante
`tests/smoke_test_environment.py`, incluyendo:

- versiones fijadas de las dependencias principales;
- NumPy, pandas, PyArrow, openpyxl y pypdf;
- Matplotlib, Seaborn y widgets;
- pydot y Graphviz;
- representación de arquitecturas con `keras.utils.plot_model()`;
- Machine Learning multisalida;
- tratamiento del desbalanceo;
- XGBoost y SHAP;
- TensorFlow/Keras multisalida;
- ejecución mediante GPU NVIDIA RTX 1000 Ada;
- guardado y restauración de modelos.

La congelación no incluye Graphviz dentro del entorno Conda, ya que se
encuentra instalado como dependencia del sistema WSL y se accede mediante
el ejecutable `/usr/bin/dot`.

A partir de esta fecha no deben incorporarse nuevas dependencias a
`TFM_ML`, salvo que sea necesario resolver una incidencia crítica para
reproducir los resultados ya obtenidos.
