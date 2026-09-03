# Reproducción del entorno TFM_Agentes

## 1. Finalidad

`TFM_Agentes` es el entorno destinado a la fase final del Trabajo Fin de
Máster sobre detección preventiva de indicadores de riesgo en salud mental
adulta mediante aprendizaje automático multisalida y agentes de inteligencia
artificial generativa a partir de NSDUH 2024.

Se utilizará para agentes, tools, salidas estructuradas, guardrails, RAG,
generación de lenguaje natural, trazabilidad y productivización mediante Flask.

La inferencia predictiva, el preprocesamiento, los modelos ML/DL, los puntos
operativos y SHAP permanecen en el entorno independiente y congelado `TFM_ML`.

## 2. Ubicaciones

Proyecto técnico en WSL:

    /home/cam/BD/TFM_NSDUH

Documentación en Windows:

    C:\Users\CAM\Documents\TFM_NSDUH

Entorno Conda:

    /home/cam/miniconda3/envs/TFM_Agentes

Kernel de Jupyter:

    Python (TFM_Agentes)

JupyterLab se ejecuta desde el entorno `base`; el Notebook 05 utilizará
el kernel `Python (TFM_Agentes)`.

## 3. Versiones principales validadas

- Python 3.11.15
- NumPy 2.4.6
- pandas 3.0.5
- requests 2.34.2
- Pydantic 2.13.4
- python-dotenv 1.2.3
- Flask 3.1.3
- Plotly 6.9.0
- pypdf 6.14.2
- ipykernel 7.3.0
- PyTorch 2.13.0
- CUDA 12.9
- sentence-transformers 5.7.0
- transformers 5.15.1
- huggingface-hub 1.28.0
- LangChain 1.3.16
- langchain-core 1.6.0
- LangGraph 1.2.11
- langchain-mistralai 1.1.6
- MCP 1.29.1
- langchain-mcp-adapters 0.3.2
- llama-index-core 0.14.23
- llama-index-embeddings-huggingface 0.7.0


MCP se mantiene en la línea 1.x mediante `mcp==1.29.1`. Esta versión se
validó conjuntamente con `langchain-mcp-adapters==0.3.2`, evitando la línea
MCP 2.x mientras no se haya comprobado su compatibilidad con la integración
LangChain/LangGraph utilizada en el TFM.

## 4. GPU validada

- NVIDIA RTX 1000 Ada Generation Laptop GPU
- VRAM: 6 GB
- PyTorch 2.13.0 compilado con CUDA 12.9
- cuDNN 9.10.2
- `torch.cuda.is_available() = True`
- Operaciones ejecutadas correctamente en `cuda:0`

Se validó `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
tanto en CPU como en GPU. En una prueba con 600 textos se obtuvieron
1.2815 s en CPU y 0.2864 s en GPU, con una aceleración de 4.48x y
resultados equivalentes.

El código deberá mantener fallback automático a CPU cuando no exista una
GPU compatible.

## 5. Reconstrucción del entorno

Desde la raíz del proyecto:

    cd "$HOME/BD/TFM_NSDUH"

Crear el entorno:

    conda env create         --file environments/TFM_Agentes_environment.yml

El archivo utiliza `conda-forge` y `nodefaults`. Su resolución ha sido
comprobada mediante `conda env create --dry-run`, reproduciendo PyTorch
2.13.0 con CUDA 12.9 y MKL.

## 6. Configuración de credenciales

Las credenciales externas se almacenan exclusivamente en:

    /home/cam/BD/TFM_NSDUH/.env

Variables utilizadas:

    MISTRAL_API_KEY
    HF_TOKEN

El archivo `.env` tiene permisos restringidos, está ignorado por Git y no
está versionado. Los valores de las credenciales no deben mostrarse en
notebooks, logs, documentación ni repositorios.

Se ha validado la autenticación de Mistral y Hugging Face sin exponer los
tokens.

## 7. Registro del kernel

El kernel registrado es:

    Python (TFM_Agentes)

Identificador:

    tfm_agentes

Comprobación:

    conda run --no-capture-output -n base         jupyter kernelspec list

El ejecutable del kernel debe ser:

    /home/cam/miniconda3/envs/TFM_Agentes/bin/python

## 8. Prueba integral

Desde la raíz del proyecto:

    conda run --no-capture-output -n TFM_Agentes         python tests/smoke_test_agents_environment.py

El test permanente deberá validar como mínimo:

- entorno y Python;
- kernel;
- NumPy, pandas y requests;
- Pydantic;
- variables de entorno seguras;
- LangChain y LangGraph;
- adaptador Mistral;
- salida estructurada;
- tool local;
- MCP Client/Server mediante transporte `stdio`;
- descubrimiento y ejecución asíncrona de tools MCP;
- adaptación de tools MCP a LangChain;
- integración MCP dentro de un `StateGraph`;
- flujo LangGraph con al menos dos nodos;
- LlamaIndex y RAG;
- sentence-transformers;
- CPU/GPU;
- Flask, HTTP y JSON;
- aislamiento respecto a `TFM_ML`;
- ausencia de dependencias rotas.


Además del smoke test inicial, se validó funcionalmente MCP el 27/08/2026
mediante un servidor local y un cliente MCP comunicados por `stdio`. El
cliente descubrió dinámicamente una tool, la ejecutó de forma asíncrona y
`langchain-mcp-adapters` permitió utilizarla desde LangChain. La misma tool
se integró posteriormente en un `StateGraph` de LangGraph, obteniendo el
resultado esperado.

El test permanente fue ejecutado satisfactoriamente el 27/08/2026, antes de iniciar el desarrollo del Notebook 05.

Resultado final: SMOKE TEST SUPERADO: True.

## 9. Apertura de JupyterLab

    bdlab "$HOME/BD/TFM_NSDUH"

Seleccionar el kernel:

    Python (TFM_Agentes)

La raíz debe ser:

    /home/cam/BD/TFM_NSDUH

## 10. Archivos de reproducibilidad

Durante el desarrollo se mantienen:

- `TFM_Agentes_environment.yml`:
  archivo de configuración con las dependencias directas necesarias.

- `TFM_Agentes_REPRODUCCION.md`:
  instrucciones y decisiones necesarias para reproducir el entorno.

- `../tests/smoke_test_agents_environment.py`:
  prueba integral permanente del entorno.

La congelación completa no se realiza todavía. Después de finalizar el
Notebook 05, ejecutar el flujo completo y confirmar que no se necesitan
más dependencias, se generarán:

- `TFM_Agentes_environment_lock.yml`
- `TFM_Agentes_requirements_lock.txt`

El entorno de agentes mantiene una separación estricta respecto a
`TFM_ML`. Se ha validado que TensorFlow, XGBoost y SHAP no están instalados
en `TFM_Agentes`, mientras continúan disponibles en `TFM_ML`.

La comunicación técnica entre ambos entornos se ha comprobado mediante
ejecución aislada y transferencia estructurada en JSON.

## 11. Comprobaciones rápidas

Dependencias:

    conda run --no-capture-output -n TFM_Agentes         python -m pip check

GPU del sistema:

    nvidia-smi

GPU desde PyTorch:

    conda run --no-capture-output -n TFM_Agentes         python -c         'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")'

Credenciales, sin mostrar valores:

    conda run --no-capture-output -n TFM_Agentes         python -c         'import os; from dotenv import load_dotenv; load_dotenv(); print(bool(os.getenv("MISTRAL_API_KEY")), bool(os.getenv("HF_TOKEN")))'

## 12. Reglas de trabajo

- Datos, notebooks y resultados técnicos se almacenan en WSL.
- No se trabaja directamente desde `/mnt/c` con datos o notebooks.
- Los documentos DOCX se almacenan en Windows.
- Los archivos originales permanecen inmutables en `data/raw`.
- `TFM_ML` permanece cerrado y congelado.
- `TFM_Agentes` no recalcula probabilidades, umbrales, clasificaciones ni SHAP.
- La capa generativa no presenta predicciones como diagnósticos ni SHAP como causalidad.
- Mistral recibe únicamente la información mínima estructurada necesaria.
- El RAG utiliza un corpus controlado y no puede alterar resultados predictivos.
- Se priorizan Pydantic, reglas deterministas, `logging`, JSON y tools explícitas.
- No se añaden dependencias sin una necesidad técnica demostrada.
- La programación se desarrolla y valida paso a paso.

## 13. Estado del entorno

Estado: **CERRADO Y CONGELADO**

El entorno ha superado las pruebas técnicas realizadas para:

- PyTorch y GPU CUDA;
- embeddings locales;
- Hugging Face;
- Mistral;
- salida estructurada con Pydantic;
- guardrails deterministas;
- tool local;
- MCP Client/Server y descubrimiento de tools;
- adaptación MCP a LangChain y LangGraph;
- LangGraph;
- LlamaIndex y RAG;
- Flask, HTTP y JSON;
- aislamiento y comunicación con `TFM_ML`;
- consistencia de dependencias.


MCP queda preparado y validado como capacidad disponible del entorno. Su uso
en la arquitectura productiva principal no se considera todavía obligatorio:
el Notebook 05 deberá comparar la tool local con la integración MCP y adoptar
la alternativa que aporte mayor utilidad sin introducir complejidad
innecesaria. `TFM_ML` permanece aislado y congelado en ambos casos.

El entorno quedó **CERRADO Y CONGELADO el 31/08/2026**, después de
finalizar el Notebook 05 y superar satisfactoriamente el smoke test integral
y la comprobación de consistencia de dependencias.

A partir de este cierre no deben instalarse, actualizarse ni eliminarse
dependencias del entorno `TFM_Agentes`. Cualquier reproducción deberá tomar
como referencia los archivos de lock definitivos:

- `TFM_Agentes_environment_lock.yml`
- `TFM_Agentes_requirements_lock.txt`

El archivo `TFM_Agentes_environment.yml` se conserva como especificación
reproducible del entorno de desarrollo, mientras que los archivos de lock
documentan el estado exacto utilizado en el cierre del TFM.
