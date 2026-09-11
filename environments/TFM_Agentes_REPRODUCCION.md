# Reproducción del entorno TFM\_Agentes

## 1\. Finalidad

`TFM\_Agentes` es el entorno destinado a la fase final del Trabajo Fin de
Máster sobre detección preventiva de indicadores de riesgo en salud mental
adulta mediante aprendizaje automático multisalida y agentes de inteligencia
artificial generativa a partir de NSDUH 2024.

Se utilizará para agentes, tools, salidas estructuradas, guardrails, RAG,
generación de lenguaje natural, trazabilidad y productivización mediante Flask.

La inferencia predictiva, el preprocesamiento, los modelos ML/DL, los puntos
operativos y SHAP permanecen en el entorno independiente y congelado `TFM\_ML`.

## 2\. Ubicaciones

Proyecto técnico en WSL:

&#x20;   /home/cam/BD/TFM\_NSDUH


Documentación en Windows:

&#x20;   C:\\Users\\CAM\\Documents\\TFM\_NSDUH


Entorno Conda:

&#x20;   /home/cam/miniconda3/envs/TFM\_Agentes


Kernel de Jupyter:

&#x20;   Python (TFM\_Agentes)


JupyterLab se ejecuta desde el entorno `base`; el Notebook 05 utilizará
el kernel `Python (TFM\_Agentes)`.

## 3\. Versiones principales validadas

* Python 3.11.15
* NumPy 2.4.6
* pandas 3.0.5
* requests 2.34.2
* Pydantic 2.13.4
* python-dotenv 1.2.3
* Flask 3.1.3
* Plotly 6.9.0
* pypdf 6.14.2
* ipykernel 7.3.0
* PyTorch 2.13.0
* CUDA 12.9
* sentence-transformers 5.7.0
* transformers 5.15.1
* huggingface-hub 1.28.0
* LangChain 1.3.16
* langchain-core 1.6.0
* LangGraph 1.2.11
* langchain-mistralai 1.1.6
* MCP 1.29.1
* langchain-mcp-adapters 0.3.2
* llama-index-core 0.14.23
* llama-index-embeddings-huggingface 0.7.0



MCP se mantiene en la línea 1.x mediante `mcp==1.29.1`. Esta versión se
validó conjuntamente con `langchain-mcp-adapters==0.3.2`, evitando la línea
MCP 2.x mientras no se haya comprobado su compatibilidad con la integración
LangChain/LangGraph utilizada en el TFM.

## 4\. GPU validada

* NVIDIA RTX 1000 Ada Generation Laptop GPU
* VRAM: 6 GB
* PyTorch 2.13.0 compilado con CUDA 12.9
* cuDNN 9.10.2
* `torch.cuda.is\_available() = True`
* Operaciones ejecutadas correctamente en `cuda:0`

Se validó `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
tanto en CPU como en GPU. En una prueba con 600 textos se obtuvieron
1.2815 s en CPU y 0.2864 s en GPU, con una aceleración de 4.48x y
resultados equivalentes.

El código deberá mantener fallback automático a CPU cuando no exista una
GPU compatible.

## 5\. Reconstrucción del entorno

Desde la raíz del proyecto:

&#x20;   cd "$HOME/BD/TFM\_NSDUH"


Crear el entorno:

&#x20;   conda env create         --file environments/TFM\_Agentes\_environment.yml


El archivo utiliza `conda-forge` y `nodefaults`. Su resolución ha sido
comprobada mediante `conda env create --dry-run`, reproduciendo PyTorch
2.13.0 con CUDA 12.9 y MKL.

## 6\. Configuración de credenciales

Las credenciales externas se almacenan exclusivamente en:

&#x20;   /home/cam/BD/TFM\_NSDUH/.env


Variables utilizadas:

&#x20;   MISTRAL\_API\_KEY
    HF\_TOKEN


El archivo `.env` tiene permisos restringidos, está ignorado por Git y no
está versionado. Los valores de las credenciales no deben mostrarse en
notebooks, logs, documentación ni repositorios.

Se ha validado la autenticación de Mistral y Hugging Face sin exponer los
tokens.

## 7\. Registro del kernel

El kernel registrado es:

&#x20;   Python (TFM\_Agentes)


Identificador:

&#x20;   tfm\_agentes


Comprobación:

&#x20;   conda run --no-capture-output -n base         jupyter kernelspec list


El ejecutable del kernel debe ser:

&#x20;   /home/cam/miniconda3/envs/TFM\_Agentes/bin/python


## 8\. Prueba integral

Desde la raíz del proyecto:

&#x20;   conda run --no-capture-output -n TFM\_Agentes         python tests/smoke\_test\_agents\_environment.py


El test permanente deberá validar como mínimo:

* entorno y Python;
* kernel;
* NumPy, pandas y requests;
* Pydantic;
* variables de entorno seguras;
* LangChain y LangGraph;
* adaptador Mistral;
* salida estructurada;
* tool local;
* MCP Client/Server mediante transporte `stdio`;
* descubrimiento y ejecución asíncrona de tools MCP;
* adaptación de tools MCP a LangChain;
* integración MCP dentro de un `StateGraph`;
* flujo LangGraph con al menos dos nodos;
* LlamaIndex y RAG;
* sentence-transformers;
* CPU/GPU;
* Flask, HTTP y JSON;
* aislamiento respecto a `TFM\_ML`;
* ausencia de dependencias rotas.



Además del smoke test inicial, se validó funcionalmente MCP el 27/08/2026
mediante un servidor local y un cliente MCP comunicados por `stdio`. El
cliente descubrió dinámicamente una tool, la ejecutó de forma asíncrona y
`langchain-mcp-adapters` permitió utilizarla desde LangChain. La misma tool
se integró posteriormente en un `StateGraph` de LangGraph, obteniendo el
resultado esperado.

El test permanente fue ejecutado satisfactoriamente el 27/08/2026, antes de iniciar el desarrollo del Notebook 05.

Resultado final: SMOKE TEST SUPERADO: True.

## 9\. Apertura de JupyterLab

&#x20;   bdlab "$HOME/BD/TFM\_NSDUH"


Seleccionar el kernel:

&#x20;   Python (TFM\_Agentes)


La raíz debe ser:

&#x20;   /home/cam/BD/TFM\_NSDUH


## 10\. Archivos de reproducibilidad

Durante el desarrollo se mantienen:

* `TFM\_Agentes\_environment.yml`:
archivo de configuración con las dependencias directas necesarias.
* `TFM\_Agentes\_REPRODUCCION.md`:
instrucciones y decisiones necesarias para reproducir el entorno.
* `../tests/smoke\_test\_agents\_environment.py`:
prueba integral permanente del entorno.

a congelación completa se realizó después de finalizar el Notebook 05, ejecutar el flujo completo

y confirmar que no se necesitaban más dependencias. Se generaron:

* `TFM\_Agentes\_environment\_lock.yml`
* `TFM\_Agentes\_requirements\_lock.txt`

El entorno de agentes mantiene una separación estricta respecto a
`TFM\_ML`. Se ha validado que TensorFlow, XGBoost y SHAP no están instalados
en `TFM\_Agentes`, mientras continúan disponibles en `TFM\_ML`.

La comunicación técnica entre ambos entornos se ha comprobado mediante
ejecución aislada y transferencia estructurada en JSON.

## 11\. Comprobaciones rápidas

Dependencias:

&#x20;   conda run --no-capture-output -n TFM\_Agentes         python -m pip check


GPU del sistema:

&#x20;   nvidia-smi


GPU desde PyTorch:

&#x20;   conda run --no-capture-output -n TFM\_Agentes         python -c         'import torch; print(torch.cuda.is\_available(), torch.cuda.get\_device\_name(0) if torch.cuda.is\_available() else "CPU")'


Credenciales, sin mostrar valores:

&#x20;   conda run --no-capture-output -n TFM\_Agentes         python -c         'import os; from dotenv import load\_dotenv; load\_dotenv(); print(bool(os.getenv("MISTRAL\_API\_KEY")), bool(os.getenv("HF\_TOKEN")))'


## 12\. Reglas de trabajo

* Datos, notebooks y resultados técnicos se almacenan en WSL.
* No se trabaja directamente desde `/mnt/c` con datos o notebooks.
* Los documentos DOCX se almacenan en Windows.
* Los archivos originales permanecen inmutables en `data/raw`.
* `TFM\_ML` permanece cerrado y congelado.
* `TFM\_Agentes` no recalcula probabilidades, umbrales, clasificaciones ni SHAP.
* La capa generativa no presenta predicciones como diagnósticos ni SHAP como causalidad.
* Mistral recibe únicamente la información mínima estructurada necesaria.
* El RAG utiliza un corpus controlado y no puede alterar resultados predictivos.
* Se priorizan Pydantic, reglas deterministas, `logging`, JSON y tools explícitas.
* No se añaden dependencias sin una necesidad técnica demostrada.
* La programación se desarrolla y valida paso a paso.

## 13\. Estado del entorno

Estado: **CERRADO Y CONGELADO**

El entorno ha superado las pruebas técnicas realizadas para:

* PyTorch y GPU CUDA;
* embeddings locales;
* Hugging Face;
* Mistral;
* salida estructurada con Pydantic;
* guardrails deterministas;
* tool local;
* MCP Client/Server y descubrimiento de tools;
* adaptación MCP a LangChain y LangGraph;
* LangGraph;
* LlamaIndex y RAG;
* Flask, HTTP y JSON;
* aislamiento y comunicación con `TFM\_ML`;
* consistencia de dependencias.



MCP quedó preparado y validado como capacidad disponible del entorno. El Notebook 05 comparó la tool

local con la integración MCP y adoptó la tool local como mecanismo principal por aportar la utilidad

necesaria sin introducir complejidad adicional.

`TFM\_ML` permanece aislado y congelado en ambos casos.

El entorno quedó **CERRADO Y CONGELADO el 31/08/2026**, después de
finalizar el Notebook 05 y superar satisfactoriamente el smoke test integral
y la comprobación de consistencia de dependencias.

A partir de este cierre no deben instalarse, actualizarse ni eliminarse
dependencias del entorno `TFM\_Agentes`. Cualquier reproducción deberá tomar
como referencia los archivos de lock definitivos:

* `TFM\_Agentes\_environment\_lock.yml`
* `TFM\_Agentes\_requirements\_lock.txt`

El archivo `TFM\_Agentes\_environment.yml` se conserva como especificación
reproducible del entorno de desarrollo, mientras que los archivos de lock
documentan el estado exacto utilizado en el cierre del TFM.
