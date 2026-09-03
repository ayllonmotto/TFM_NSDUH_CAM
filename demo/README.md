# Dashboard demo — TFM NSDUH 2024

**Autor:** Carlos Ayllón Motto
**Trabajo Fin de Máster:** Sistema explicable de detección preventiva de indicadores de riesgo en salud mental adulta mediante aprendizaje automático multisalida y agentes de inteligencia artificial generativa a partir del dataset NSDUH 2024
**Versión:** 1.0.0


Interfaz local de demostración del sistema desarrollado en el Trabajo Fin de Máster:

Sistema explicable de detección preventiva de indicadores de riesgo en salud mental adulta mediante aprendizaje automático multisalida y agentes de inteligencia artificial generativa a partir del dataset NSDUH 2024.

La demo utiliza el runtime real del proyecto.

No reentrena modelos, no recalibra umbrales y no vuelve a utilizar el conjunto TEST.

## 1\. Ejecución recomendada

### Desde WSL

&#x20;   cd \~/BD/TFM\_NSDUH/demo
    bash iniciar\_dashboard\_demo.sh


### Desde Windows mediante WSL

Ejecutar:

&#x20;   INICIAR\_DASHBOARD\_DEMO.cmd


El dashboard se sirve localmente en:

&#x20;   http://127.0.0.1:5000/dashboard


La aplicación real se inicializa en segundo plano mediante el entorno `TFM\_Agentes`.

## 2\. Componentes

* `dashboard\_demo.py`: capa de presentación y navegación.
* `dashboard\_demo.html`: interfaz visual del dashboard.
* `dashboard\_assets/`: figuras estáticas exportadas de los notebooks y captura de la API.
* `demo\_tfm.py`: runtime base de la aplicación Flask real.
* `iniciar\_dashboard\_demo.sh`: lanzador principal desde WSL.
* `INICIAR\_DASHBOARD\_DEMO.cmd`: lanzador desde Windows mediante WSL.
* `iniciar\_demo.sh`: acceso directo opcional a la aplicación base.

## 3\. Menú

### Demo

* Resumen
* Aplicación
* Historial

### Modelo

* Modelos
* Explicabilidad SHAP
* Evidencias
* Trazabilidad

### Sistema

* Arquitectura
* API
* Acerca de

## 4\. Modelos

La sección `Modelos` permite explorar de forma interactiva la solución híbrida final:

|Variable|Familia|Modelo|Umbral|
|-|-|-|-:|
|IRAMDEYR|ML|XGBoost|0,500|
|IRSUICTHNK|DL|Ensemble|0,509|
|IRSUIPLANYR|DL|Ensemble|0,536|
|IRSUITRYYR|DL|Ensemble|0,502|

La vista permite:

* seleccionar cualquiera de las cuatro variables objetivo;
* filtrar por familia ML o DL;
* consultar el modelo final asociado;
* visualizar el umbral de clasificación;
* comparar las probabilidades disponibles de `demo\_01`, `demo\_02` y `demo\_03` frente a ese umbral.

La interacción es exclusivamente exploratoria.

No modifica los modelos, los hiperparámetros, los umbrales ni los resultados científicos ya cerrados.

## 5\. Trazabilidad

El dashboard representa el recorrido funcional del sistema:

&#x20;   Entrada CSV
        ↓
    Validación Pydantic
        ↓
    Tool predictiva local
        ↓
    TFM\_ML
        ↓
    ML / DL
        ↓
    SHAP
        ↓
    RAG
        ↓
    Mistral + LangGraph
        ↓
    Guardrails
        ↓
    Informe e historial


La inferencia predictiva y la explicación matemática permanecen separadas de la capa generativa.

La vista ampliada recoge datos del contrato y de las ejecuciones registradas, incluyendo 813 variables de entrada, 4 predicciones, 5 nodos operativos, 13 consultas documentales, 15 variables SHAP principales por objetivo, 12 factores autorizados para la comunicación generativa y 37.840 predicciones de TEST conservadas como referencia.

## 6\. Explicabilidad SHAP y evidencias

`Explicabilidad SHAP` presenta las figuras estáticas exportadas del Notebook 04: importancia global, beeswarm por objetivo y waterfall de casos individuales. También permite consultar el detalle local `detalle\_explicabilidad` del informe JSON seleccionado.

`Evidencias` reúne figuras estáticas de EDA, ML, DL, comparación de familias, curvas Precision–Recall, estabilidad del recall frente al umbral, matrices de confusión e interfaz API. Las imágenes no se recalculan desde el dashboard.

## 7\. Arquitectura de entornos

### TFM\_ML

Responsable de:

* preprocesamiento;
* inferencia Machine Learning;
* inferencia Deep Learning;
* probabilidades;
* aplicación de umbrales;
* explicaciones SHAP.

### TFM\_Agentes

Responsable de:

* validación estructurada con Pydantic;
* tool predictiva;
* LangGraph;
* recuperación documental RAG;
* generación mediante Mistral;
* guardrails;
* aplicación Flask;
* historial de informes;
* ejecución del dashboard.

La comunicación entre ambos conserva la arquitectura de productivización definida en el Notebook 05.

## 8\. Aplicación

La opción `Aplicación` utiliza la pantalla de acceso real del prototipo desarrollado en el Notebook 05.

Credencial académica de demostración:

&#x20;   CAM


Desde la aplicación pueden utilizarse los casos de demostración y generarse informes mediante el flujo real del sistema.

## 9\. Historial

La sección `Historial` recupera los informes guardados en el directorio definido por la aplicación.

Los casos canónicos de demostración son:

* `demo\_01`
* `demo\_02`
* `demo\_03`

## 10\. API

El dashboard permite consultar la documentación de los endpoints implementados por la aplicación Flask.

La API forma parte de la misma arquitectura utilizada por el prototipo académico y no constituye un servicio clínico de producción.

## 11\. Uso responsable

Este sistema es un prototipo académico orientado a la detección preventiva de indicadores de riesgo.

No constituye un diagnóstico clínico.

No sustituye la valoración de profesionales sanitarios.

No debe utilizarse para tomar decisiones clínicas automatizadas.

Las predicciones y explicaciones deben interpretarse exclusivamente en el contexto académico y experimental para el que se ha desarrollado el proyecto.

## 12\. Cierre

El botón `Cerrar demo` detiene el servidor local.

También puede utilizarse `Ctrl+C` desde la terminal desde la que se inició el dashboard.

## 13\. Contenido final

* Panel lateral con consulta en directo de `/api/v1/health`.
* Enlace visible al repositorio público: https://github.com/ayllonmotto/TFM\_NSDUH\_CAM
* Explicabilidad SHAP con figuras estáticas exportadas del Notebook 04 y detalle local recuperado de los informes JSON.
* Evidencias visuales estáticas de EDA, ML, DL, evaluación, umbrales, matrices de confusión y API.
* Ampliación de figuras al hacer clic para facilitar la presentación en vídeo.
* Detalle de informes ampliado con predicciones, umbrales, clasificación y trazabilidad.
* Navegación homogénea entre resumen, aplicación, historial, modelos, SHAP, arquitectura y API.
* Unificación tipográfica y de tamaños entre textos, controles y botones.
* Trazabilidad ampliada con datos registrados del contrato API, SHAP, documentación y predicciones de referencia.
* Sección `Acerca de` con autoría y repositorio público.

El dashboard esa una capa de presentación. No modifica notebooks, modelos, umbrales, explicaciones SHAP ni resultados científicos.
