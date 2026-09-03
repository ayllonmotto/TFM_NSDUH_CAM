# Model Card — TFM NSDUH 2024

## Sistema explicable de detección preventiva de indicadores de riesgo en salud mental adulta mediante aprendizaje automático multisalida y agentes de inteligencia artificial generativa a partir del dataset NSDUH 2024

### Explainable system for preventive detection of risk indicators in adult mental health using multi-output machine learning and generative AI agents based on the NSDUH 2024 dataset

## Resumen / Abstract

Este trabajo presenta un prototipo académico explicable para la detección preventiva de cuatro indicadores relacionados con salud mental adulta a partir de NSDUH 2024. La solución combina Machine Learning y Deep Learning, explicabilidad mediante SHAP y una capa de agentes de inteligencia artificial generativa con RAG, LangGraph, Mistral y guardrails.

This work presents an explainable academic prototype for the preventive detection of four indicators related to adult mental health using NSDUH 2024. The system combines Machine Learning and Deep Learning, SHAP-based explainability, and a generative AI agent layer integrating RAG, LangGraph, Mistral, and guardrails. The project includes reproducible notebooks, experimental results, final inference files, and a local demonstration dashboard.

## 1. Identificación / Identification

**Proyecto / Project:** Sistema explicable de detección preventiva de indicadores de riesgo en salud mental adulta mediante aprendizaje automático multisalida y agentes de inteligencia artificial generativa a partir del dataset NSDUH 2024.

**English title:** Explainable system for preventive detection of risk indicators in adult mental health using multi-output machine learning and generative AI agents based on the NSDUH 2024 dataset.

**Tipo de sistema / System type:** prototipo académico de clasificación multisalida con solución híbrida Machine Learning / Deep Learning y capa de explicabilidad.

**Versión predictiva / Predictive version:** v1.

**Fuente de datos / Data source:** National Survey on Drug Use and Health (NSDUH) 2024.

---

## 2. Finalidad / Intended purpose

El sistema se ha desarrollado para estudiar la viabilidad de detectar de forma preventiva determinados indicadores relacionados con salud mental adulta a partir de variables disponibles en NSDUH 2024.

El objetivo es académico y experimental.

El sistema:

- no constituye un diagnóstico clínico;
- no sustituye la valoración de profesionales sanitarios;
- no debe utilizarse para tomar decisiones clínicas automatizadas;
- no ha sido validado como dispositivo médico ni como herramienta asistencial.

---

## 3. Población de estudio / Study population

El proyecto trabaja con población adulta de 18 años o más procedente de NSDUH 2024.

El archivo original utilizado durante el desarrollo contiene:

- 58.633 registros;
- 2.632 variables.

El dataset original no se distribuye con el repositorio.

---

## 4. Variables objetivo / Target variables

El sistema genera cuatro salidas binarias:

| Variable | Indicador |
| --- | --- |
| `IRAMDEYR` | Episodio depresivo mayor |
| `IRSUICTHNK` | Ideación suicida |
| `IRSUIPLANYR` | Planificación suicida |
| `IRSUITRYYR` | Intento suicida |

---

## 5. Solución final / Final solution

La evaluación final dio lugar a una solución híbrida:

| Variable | Familia | Modelo final | Umbral |
| --- | --- | --- | ---: |
| `IRAMDEYR` | ML | XGBoost | 0,500 |
| `IRSUICTHNK` | DL | Ensemble multisalida | 0,509 |
| `IRSUIPLANYR` | DL | Ensemble multisalida | 0,536 |
| `IRSUITRYYR` | DL | Ensemble multisalida | 0,502 |

El ensemble Deep Learning combina con ponderación 50/50:

- `multisalida_focal_ramas_32`;
- `multisalida_focal_sin_bn_ramas`.

La selección de familia, modelos y umbrales forma parte del cierre científico del proyecto y no se modifica en la capa de demostración.

---

## 6. Archivos necesarios para inferencia / Inference files

### Machine Learning

    models/02_modelado_ml/modelos_ml_finales.pkl

### Deep Learning

    models/03_modelado_dl/preprocesamiento_final_dl_v1.joblib
    models/03_modelado_dl/metadatos_modelo_final_dl_v1.joblib
    models/03_modelado_dl/modelo_final_focal_ramas_32_v1.keras
    models/03_modelado_dl/modelo_final_focal_sin_bn_ramas_v1.keras

### Transferencia y explicabilidad

    results/tables/04_evaluacion_explicabilidad/configuracion_productivizacion_v1.joblib
    results/tables/04_evaluacion_explicabilidad/background_shap_productivizacion_v1.joblib
    results/tables/04_evaluacion_explicabilidad/77_limitaciones_modelo_final.csv

Los archivos anteriores constituyen el conjunto mínimo necesario para ejecutar la solución predictiva y explicativa final.

---

## 7. Entradas y preprocesamiento / Inputs and preprocessing

El esquema final de entrada procede de las variables seleccionadas durante las fases de EDA, modelado ML y modelado DL.

El servicio de productivización recupera la configuración definitiva desde los archivos generados durante la evaluación final.

El preprocesamiento Deep Learning se encuentra guardado en:

    preprocesamiento_final_dl_v1.joblib

Este archivo contiene, entre otros componentes, la información necesaria para transformar las variables numéricas y categóricas utilizadas por los modelos DL finales.

El esquema detallado de productivización puede consultarse en:

    results/tables/04_evaluacion_explicabilidad/75_esquema_entrada_productivizacion.csv
    results/tables/04_evaluacion_explicabilidad/76_esquema_salida_productivizacion.csv

---

## 8. Salidas / Outputs

Para cada una de las cuatro variables objetivo, el servicio predictivo devuelve información estructurada que incluye:

- probabilidad estimada;
- umbral de decisión;
- clasificación binaria;
- familia de modelo utilizada;
- modelo seleccionado.

La capa posterior de explicabilidad añade información SHAP para interpretar la contribución de las variables a la predicción.

---

## 9. Evaluación / Evaluation

La evaluación final se realizó en el Notebook 04 sobre el conjunto TEST reservado durante el desarrollo.

Las métricas utilizadas incluyen, según la fase de análisis:

- Average Precision;
- ROC-AUC;
- Recall;
- Precision;
- F1 / F2;
- Balanced Accuracy;
- Specificity;
- análisis de falsos negativos y falsos positivos;
- estabilidad frente al umbral;
- comparación ML frente a DL.

Los resultados cuantitativos definitivos se conservan en los archivos generados por el propio proyecto, especialmente:

    results/tables/04_evaluacion_explicabilidad/17_metricas_finales_test.csv
    results/tables/04_evaluacion_explicabilidad/25_diferencias_metricas_ml_dl_test.csv
    results/tables/04_evaluacion_explicabilidad/37_resumen_global_metricas_test.csv
    results/tables/04_evaluacion_explicabilidad/51_intervalos_bootstrap_ml_dl_test.csv
    results/tables/04_evaluacion_explicabilidad/53_seleccion_final_por_objetivo.csv

La demo no recalcula estas métricas y no reutiliza TEST para seleccionar o modificar modelos.

---

## 10. Explicabilidad / Explainability

La solución incorpora explicabilidad mediante SHAP.

El análisis incluye:

- importancia global;
- explicaciones locales;
- gráficos beeswarm;
- gráficos de dependencia;
- casos individuales mediante waterfall;
- comparación de variables relevantes entre objetivos.

Las evidencias se encuentran en:

    results/figures/04_evaluacion_explicabilidad/
    results/tables/04_evaluacion_explicabilidad/

La explicación matemática producida mediante SHAP se mantiene separada de la capa generativa.

---

## 11. Capa generativa / Generative layer

La solución de productivización incorpora una capa de agentes desarrollada en el Notebook 05.

Esta capa utiliza:

- contratos estructurados con Pydantic;
- tool predictiva local;
- LangGraph;
- recuperación documental RAG;
- generación mediante Mistral;
- guardrails;
- aplicación Flask;
- generación e historial de informes.

La capa generativa no sustituye las probabilidades producidas por los modelos predictivos ni modifica sus umbrales.

---

## 12. Casos de demostración / Demonstration cases

El proyecto conserva tres casos canónicos:

    demo_01
    demo_02
    demo_03

Estos casos muestran diferentes niveles de señal generada por la solución final.

No deben interpretarse como una representación de la prevalencia o distribución de la población NSDUH.

---

## 13. Limitaciones / Limitations

Entre las principales limitaciones del sistema se encuentran:

- uso de una única fuente de datos observacional;
- dependencia de la población y del cuestionario NSDUH 2024;
- desbalanceo de las variables objetivo;
- posible variabilidad en el rendimiento entre subgrupos;
- ausencia de validación clínica externa;
- ausencia de validación prospectiva;
- dependencia del esquema de entrada establecido durante el desarrollo;
- sensibilidad inherente a la selección de umbrales;
- dependencia de librerías y archivos serializados;
- variabilidad de la capa generativa.

Las limitaciones específicas documentadas durante la evaluación están disponibles en:

    results/tables/04_evaluacion_explicabilidad/77_limitaciones_modelo_final.csv

---

## 14. Uso previsto / Intended use

Usos compatibles con el alcance del proyecto:

- investigación académica;
- demostración de arquitecturas ML/DL multisalida;
- estudio de explicabilidad;
- experimentación con productivización;
- demostración de integración entre modelos predictivos y agentes generativos;
- reproducción técnica del Trabajo Fin de Máster.

---

## 15. Usos no previstos / Out-of-scope uses

El sistema no está diseñado para:

- realizar diagnósticos de salud mental;
- sustituir entrevistas o evaluaciones clínicas;
- establecer tratamientos;
- realizar triaje clínico automatizado;
- tomar decisiones administrativas, laborales, aseguradoras o legales sobre personas;
- desplegarse directamente en producción sanitaria sin validaciones adicionales.

---

## 16. Consideraciones de interpretación / Interpretation

Una clasificación positiva significa que la probabilidad generada por el modelo ha superado el umbral definido para esa variable objetivo.

No significa que exista un diagnóstico clínico.

Una clasificación negativa tampoco permite descartar clínicamente la existencia de un problema.

Las predicciones deben interpretarse teniendo en cuenta las limitaciones estadísticas, metodológicas y de población del proyecto.

---

## 17. Reproducibilidad / Reproducibility

Los entornos utilizados están documentados en:

    environments/TFM_ML_REPRODUCCION.md
    environments/TFM_Agentes_REPRODUCCION.md

El repositorio incluye además:

- archivos Conda `environment.yml`;
- archivos de bloqueo;
- requirements;
- pruebas de entorno;
- notebooks ejecutados;
- tablas y figuras de resultados;
- archivos finales de inferencia.

La reproducción completa de las fases experimentales requiere obtener NSDUH 2024 desde su fuente oficial.

---

## 18. Demo / Demonstration

La interfaz final se encuentra en:

    demo/

La documentación específica está disponible en:

    demo/README.md

El dashboard permite explorar la arquitectura y ejecutar el prototipo sin modificar el cierre científico del proyecto.

---

## 19. Trazabilidad científica / Scientific traceability

La secuencia principal del proyecto es:

    01 Datos y EDA
        ↓
    02 Modelado ML
        ↓
    03 Modelado DL
        ↓
    04 Evaluación y explicabilidad
        ↓
    05 Agentes y productivización
        ↓
    Dashboard de demostración

Los notebooks 01–05 constituyen la referencia principal para reconstruir las decisiones metodológicas y técnicas.

---

## 20. Estado / Status

Modelo y arquitectura predictiva: **cerrados**.

Evaluación final: **cerrada**.

Explicabilidad: **cerrada**.

Productivización: **cerrada**.

Dashboard local: **validado**.

El documento describe la versión preparada para publicación académica del proyecto.
