# TFM NSDUH 2024

## Sistema explicable de detección preventiva de indicadores de riesgo en salud mental adulta mediante aprendizaje automático multisalida y agentes de inteligencia artificial generativa a partir del dataset NSDUH 2024

### Explainable system for preventive detection of risk indicators in adult mental health using multi-output machine learning and generative AI agents based on the NSDUH 2024 dataset

Trabajo Fin de Máster desarrollado sobre la encuesta **National Survey on Drug Use and Health (NSDUH) 2024**.

## Resumen / Abstract

Este proyecto desarrolla un prototipo académico explicable para la detección preventiva de cuatro indicadores relacionados con salud mental adulta a partir de NSDUH 2024. La solución final combina Machine Learning y Deep Learning, explicabilidad mediante SHAP y una capa de agentes de inteligencia artificial generativa con RAG, LangGraph, Mistral y guardrails. El repositorio incluye los cinco notebooks principales, resultados experimentales, archivos finales de inferencia, documentación de reproducibilidad y un dashboard local de demostración.

This project presents an explainable academic prototype for the preventive detection of four indicators related to adult mental health using NSDUH 2024. The final system combines Machine Learning and Deep Learning, SHAP-based explainability, and a generative AI agent layer integrating RAG, LangGraph, Mistral, and guardrails. The repository includes the five main project notebooks, experimental results, final inference files, reproducibility documentation, and a local demonstration dashboard.

> \\\*\\\*Uso académico y preventivo / Academic and preventive use.\\\*\\\* Este proyecto no constituye un sistema de diagnóstico clínico, no sustituye la valoración de profesionales sanitarios y no debe utilizarse para tomar decisiones clínicas automatizadas. This project is not a clinical diagnostic system, does not replace professional health assessment, and must not be used for automated clinical decision-making.

\---

## 1\. Variables objetivo

El modelado final considera cuatro salidas:

|Variable|Indicador|
|-|-|
|`IRAMDEYR`|Episodio depresivo mayor|
|`IRSUICTHNK`|Ideación suicida|
|`IRSUIPLANYR`|Planificación suicida|
|`IRSUITRYYR`|Intento suicida|

La selección final combina Machine Learning y Deep Learning según el comportamiento observado en la evaluación final sobre TEST.

\---

## 2\. Solución predictiva final

La arquitectura productivizada utiliza una solución híbrida:

|Variable|Familia final|Modelo|Umbral|
|-|-|-|-:|
|`IRAMDEYR`|ML|XGBoost|0,500|
|`IRSUICTHNK`|DL|Ensemble multisalida|0,509|
|`IRSUIPLANYR`|DL|Ensemble multisalida|0,536|
|`IRSUITRYYR`|DL|Ensemble multisalida|0,502|

El ensemble DL combina los modelos finales:

* `multisalida\\\_focal\\\_ramas\\\_32`;
* `multisalida\\\_focal\\\_sin\\\_bn\\\_ramas`.

Los modelos, umbrales y decisiones científicas incluidos en esta versión corresponden al cierre definitivo del proyecto.

\---

## 3\. Flujo general

El proyecto está organizado en cinco notebooks principales:

1. `01\\\_TFM\\\_Datos\\\_EDA.ipynb`
Preparación de datos, análisis exploratorio y definición del conjunto de variables.
2. `02\\\_TFM\\\_Modelado\\\_ML.ipynb`
Modelado Machine Learning, selección de variables, ajuste y comparación de modelos.
3. `03\\\_TFM\\\_Modelado\\\_DL.ipynb`
Desarrollo Deep Learning, análisis de arquitecturas, ensembles y selección final.
4. `04\\\_TFM\\\_Evaluacion\\\_Explicabilidad.ipynb`
Evaluación final sobre TEST, comparación ML/DL, selección híbrida y explicabilidad SHAP.
5. `05\\\_TFM\\\_Agentes\\\_Productivizacion.ipynb`
Productivización, contratos estructurados, tools, RAG, LangGraph, generación mediante Mistral, guardrails, API y evaluación extremo a extremo.

\---

## 4\. Arquitectura de productivización

La solución separa dos entornos principales.

### TFM\_ML

Responsable de:

* preprocesamiento;
* inferencia Machine Learning;
* inferencia Deep Learning;
* cálculo de probabilidades;
* aplicación de umbrales;
* explicaciones SHAP.

### TFM\_Agentes

Responsable de:

* validación estructurada con Pydantic;
* tool predictiva;
* comunicación con `TFM\\\_ML`;
* recuperación documental RAG;
* orquestación mediante LangGraph;
* generación mediante Mistral;
* guardrails;
* API Flask;
* generación e historial de informes;
* dashboard de demostración.

La separación permite mantener las dependencias de inferencia predictiva aisladas de la capa de agentes.

\---

## 5\. Dashboard de demostración

La carpeta `demo/` contiene la interfaz local final del proyecto.

Ejecución desde WSL:

&#x20;   cd \~/BD/TFM\_NSDUH/demo

&#x20;   bash iniciar\_dashboard\_demo.sh



Ejecución desde Windows mediante WSL:

&#x20;   demo\\INICIAR\_DASHBOARD\_DEMO.cmd



URL local:

&#x20;   http://127.0.0.1:5000/dashboard



La demo proporciona acceso a:

* resumen del proyecto;
* aplicación predictiva real;
* historial de informes;
* exploración interactiva de los modelos finales;
* trazabilidad del sistema;
* arquitectura;
* documentación de API;
* información técnica del prototipo.

La documentación específica se encuentra en:

&#x20;   demo/README.md



\---

## 6\. Estructura del repositorio

&#x20;   TFM\_NSDUH\_CAM/
├── demo/
├── environments/
├── models/
│   ├── 02\_modelado\_ml/
│   └── 03\_modelado\_dl/
├── notebooks/
├── results/
│   ├── agent\_reports/
│   ├── figures/
│   └── tables/
├── src/
│   └── productivizacion/
├── tests/
├── .gitignore
├── CITATION.cff
├── LICENSE
├── MODEL\_CARD.md
├── NOTICE.md
└── README.md



Esta estructura corresponde al repositorio público de entrega. Conserva los notebooks, resultados, figuras, tablas, modelos finales, código de productivización y documentación necesarios para revisar el proyecto.

El dataset NSDUH 2024, las credenciales, los checkpoints intermedios, las cachés y otros archivos locales no necesarios para la distribución final no forman parte del repositorio público.

\---

## 7\. Archivos finales de inferencia

La versión publicada incorpora únicamente los modelos necesarios para ejecutar la solución final:

&#x20;   models/02\_modelado\_ml/modelos\_ml\_finales.pkl

&#x20;   models/03\\\_modelado\\\_dl/preprocesamiento\\\_final\\\_dl\\\_v1.joblib
    models/03\\\_modelado\\\_dl/metadatos\\\_modelo\\\_final\\\_dl\\\_v1.joblib
    models/03\\\_modelado\\\_dl/modelo\\\_final\\\_focal\\\_ramas\\\_32\\\_v1.keras
    models/03\\\_modelado\\\_dl/modelo\\\_final\\\_focal\\\_sin\\\_bn\\\_ramas\\\_v1.keras





Los archivos de transferencia utilizados por la capa de productivización se encuentran en:

&#x20;   results/tables/04\_evaluacion\_explicabilidad/



Entre ellos:

* `configuracion\\\_productivizacion\\\_v1.joblib`;
* `background\\\_shap\\\_productivizacion\\\_v1.joblib`;
* `77\\\_limitaciones\\\_modelo\\\_final.csv`.

\---

## 8\. Dataset

El proyecto utiliza como fuente:

**NSDUH 2024 — National Survey on Drug Use and Health**

Archivo utilizado durante el desarrollo:

&#x20;   data/raw/NSDUH\_2024\_Tab.txt



Características verificadas:

* formato TSV con extensión `.txt`;
* 58.633 registros;
* 2.632 variables;
* 58.634 líneas físicas incluida la cabecera;
* tamaño: 390.147.105 bytes;
* SHA-256:

03276012b24533bb6d6a5fc7978c21bfe35d73ffeb406e731eafd3e9bc18b650

* Fuente oficial de descarga y documentación:

https://www.samhsa.gov/data/data-we-collect/nsduh-national-survey-drug-use-and-health/datafiles



El dataset original **no se distribuye dentro del repositorio** y permanece excluido mediante `.gitignore`.

\---

## 9\. Entornos de ejecución

El proyecto utiliza dos entornos Conda:

&#x20;   TFM\_ML
TFM\_Agentes



Versiones principales de la capa predictiva:

* Python 3.11.15
* scikit-learn 1.9.0
* imbalanced-learn 0.14.2
* XGBoost 3.2.0
* SHAP 0.51.0
* TensorFlow 2.20.0
* Keras 3.15.1

GPU utilizada y validada durante el desarrollo:

&#x20;   NVIDIA RTX 1000 Ada Generation Laptop GPU



Los archivos de reproducción están disponibles en:

&#x20;   environments/



Documentos principales:

* `TFM\\\_ML\\\_REPRODUCCION.md`
* `TFM\\\_Agentes\\\_REPRODUCCION.md`

También se incluyen los archivos `environment.yml`, locks y requirements correspondientes.

\---

## 10\. Código de productivización

La implementación reutilizable se encuentra en:

&#x20;   src/productivizacion/



Componentes principales:

* `servicio\\\_predictivo.py`: servicio de inferencia y explicabilidad ejecutado mediante `TFM\\\_ML`;
* `servidor\\\_mcp\\\_predictivo.py`: servidor MCP desarrollado y evaluado durante el proyecto.

La solución final utiliza el mecanismo de tool local seleccionado en la evaluación del Notebook 05, manteniéndose MCP como parte de la implementación y comparación realizada.

\---

## 11\. Resultados y trazabilidad

Los resultados del proyecto están organizados por etapa en:

&#x20;   results/figures/
results/tables/
results/agent\_reports/



Se incluyen, entre otros:

* análisis exploratorio;
* resultados de validación cruzada;
* comparación ML;
* experimentación DL;
* evaluación final sobre TEST;
* curvas y matrices de confusión;
* análisis SHAP;
* tablas de cierre;
* evaluación de tools;
* resultados RAG;
* trazabilidad LangGraph;
* pruebas de guardrails;
* evaluación extremo a extremo;
* informes de demostración.

Los tres casos utilizados para la demostración final son:

&#x20;   demo\_01
demo\_02
demo\_03



Estos casos permiten mostrar distintos niveles de señal predictiva sin pretender representar la distribución de la población NSDUH.

\---

## 12\. Reproducibilidad

El repositorio público se distribuye con el nombre:

&#x20;   TFM\_NSDUH\_CAM



La implementación final fue desarrollada y validada en WSL utilizando como ubicación local de referencia:

&#x20;   \~/BD/TFM\_NSDUH



Para reproducir la ejecución sin modificar las rutas internas de la versión validada, se recomienda situar el contenido del repositorio público en esa ubicación local.

Las instrucciones detalladas de reconstrucción de los entornos están disponibles en la carpeta `environments/`.

La reproducción completa de las fases experimentales puede requerir el dataset NSDUH 2024 original, que debe obtenerse desde su fuente oficial y situarse en la ruta definida por el proyecto.

La ejecución de la demo final requiere además configurar las credenciales necesarias mediante variables de entorno. Los archivos `.env` y las credenciales no forman parte del repositorio.

\---

## 13\. Alcance y limitaciones

El sistema debe interpretarse como un prototipo académico de apoyo preventivo.

Entre sus principales limitaciones se encuentran:

* dependencia de la población y variables disponibles en NSDUH 2024;
* naturaleza observacional de la fuente de datos;
* desbalanceo de las variables objetivo;
* ausencia de validación clínica externa;
* dependencia del esquema de entrada definido durante el proyecto;
* necesidad de supervisión humana para interpretar resultados;
* variabilidad inherente a la generación mediante modelos de lenguaje.

Las probabilidades generadas no deben interpretarse como diagnósticos ni como estimaciones clínicas individuales.

\---

## 14\. Estado del proyecto

La versión actual corresponde al cierre técnico del TFM:

* EDA finalizado;
* modelado ML finalizado;
* modelado DL finalizado;
* evaluación final y explicabilidad finalizadas;
* solución híbrida seleccionada;
* productivización finalizada;
* agentes y RAG integrados;
* evaluación extremo a extremo completada;
* dashboard local validado;
* archivos finales de inferencia preparados para publicación.

La documentación pública, versión de release y registro de preservación se completan como parte del proceso final de publicación del proyecto.

\---

## 15\. Citación / Citation

La información recomendada para citar este proyecto se encuentra en:

* [`CITATION.cff`](CITATION.cff)

GitHub puede utilizar este archivo para mostrar automáticamente la información de citación del repositorio.

La versión preparada para la primera publicación estable es:

&#x20;   v1.0.0



La versión `v1.0.0` está archivada en Zenodo con el **DOI específico de versión** [`10.5281/zenodo.22277709`](https://doi.org/10.5281/zenodo.22277709). El **Concept DOI** [`10.5281/zenodo.22277708`](https://doi.org/10.5281/zenodo.22277708) identifica el conjunto de versiones del proyecto.

**Nota de publicación.** La rama `main` se actualizó después de la asignación del DOI exclusivamente para incorporar los identificadores de Zenodo a `README.md` y `CITATION.cff`. Esta actualización es únicamente documental y no modifica la release `v1.0.0` ni el registro de dicha versión archivado en Zenodo.

\---

## 16\. Licencia / License

El software original de este repositorio se distribuye bajo la:

**MIT License**

Véase:

* [`LICENSE`](LICENSE)
* [`NOTICE.md`](NOTICE.md)

El dataset original NSDUH 2024 no se distribuye en este repositorio y no queda cubierto por la licencia MIT del proyecto.

Los datos, publicaciones oficiales, marcas y demás materiales de terceros permanecen sujetos a sus respectivas condiciones de uso.

La ficha técnica y las limitaciones del sistema se documentan en:

* [`MODEL\\\_CARD.md`](MODEL_CARD.md)

La documentación específica del dashboard se encuentra en:

* [`demo/README.md`](demo/README.md)
