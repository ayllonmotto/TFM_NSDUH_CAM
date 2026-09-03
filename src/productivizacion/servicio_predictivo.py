# =============================================================================
# SERVICIO PREDICTIVO — TFM_ML
# =============================================================================
import json
import os
import pickle
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from pathlib import Path
from time import perf_counter

import joblib
import keras
import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURACIÓN Y RUTAS
# =============================================================================
ruta_proyecto = Path.home() / 'BD' / 'TFM_NSDUH'

ruta_modelos_ml = ruta_proyecto / 'models' / '02_modelado_ml'
ruta_modelos_dl = ruta_proyecto / 'models' / '03_modelado_dl'
ruta_tablas_eda = ruta_proyecto / 'results' / 'tables' / '01_datos_eda'
ruta_transferencia = ruta_proyecto / 'results' / 'tables' / '04_evaluacion_explicabilidad'

ruta_pickle_ml = ruta_modelos_ml / 'modelos_ml_finales.pkl'
ruta_preprocesamiento_dl = ruta_modelos_dl / 'preprocesamiento_final_dl_v1.joblib'
ruta_metadatos_dl = ruta_modelos_dl / 'metadatos_modelo_final_dl_v1.joblib'
ruta_roles = ruta_tablas_eda / '28_roles_base_premodelado.csv'
ruta_numericas = ruta_tablas_eda / '30_resumen_numericas_validas.csv'

ruta_configuracion = ruta_transferencia / 'configuracion_productivizacion_v1.joblib'
ruta_background = ruta_transferencia / 'background_shap_productivizacion_v1.joblib'
ruta_limitaciones = ruta_transferencia / '77_limitaciones_modelo_final.csv'

ruta_base_premodelado = (
    ruta_proyecto / 'data' / 'interim' / 'NSDUH_2024_adultos_premodelado.parquet'
)

encoding_csv = 'utf-8-sig'
n_repeticiones_inferencia = 10


# =============================================================================
# RECUPERACIÓN DE ARCHIVOS
# =============================================================================
configuracion = joblib.load(ruta_configuracion)
background = joblib.load(ruta_background)

with ruta_pickle_ml.open('rb') as archivo:
    pickle_ml = pickle.load(archivo)

preprocesamiento_dl = joblib.load(ruta_preprocesamiento_dl)
metadatos_dl = joblib.load(ruta_metadatos_dl)

rangos_numericas = pd.read_csv(
    ruta_numericas, encoding=encoding_csv
)[['variable', 'limite_inferior', 'limite_superior']].copy()

limitaciones = pd.read_csv(ruta_limitaciones, encoding=encoding_csv)

roles_base = pd.read_csv(ruta_roles, encoding=encoding_csv)

etiquetas_variables = (
    roles_base.dropna(subset=['etiqueta_codebook']).drop_duplicates('variable')
    .set_index('variable')['etiqueta_codebook'].to_dict()
)

diccionario_rangos = rangos_numericas.set_index('variable')[
    ['limite_inferior', 'limite_superior']
].to_dict('index')


# -----------------------------------------------------------------------------
# MACHINE LEARNING
# -----------------------------------------------------------------------------
modelos_ml = pickle_ml['modelos']

vars_predictoras_modelado_ml = list(pickle_ml['variables_originales'])
vars_numericas_modelado_ml = list(pickle_ml['variables_numericas'])
vars_categoricas_modelado_ml = list(pickle_ml['variables_categoricas'])

vars_estado_modelado_ml = [
    variable for variable in vars_categoricas_modelado_ml if variable.endswith('_estado')
]

vars_categoricas_originales_modelado_ml = [
    variable for variable in vars_categoricas_modelado_ml if not variable.endswith('_estado')
]

columnas_entrada_ml = list(pickle_ml['columnas_entrada'])


# -----------------------------------------------------------------------------
# DEEP LEARNING
# -----------------------------------------------------------------------------
targets = list(configuracion['targets'])
titulos_targets = dict(configuracion['titulos_targets'])
familias_finales = dict(configuracion['familias_finales'])
modelos_finales = dict(configuracion['modelos_finales'])
umbrales = dict(configuracion['umbrales'])

vars_predictoras_modelado_dl = list(configuracion['variables_dl'])

vars_numericas_modelado_dl = list(preprocesamiento_dl['variables_numericas'])
vars_categoricas_modelado_dl = list(preprocesamiento_dl['variables_categoricas'])

vars_estado_modelado_dl = [
    variable for variable in vars_categoricas_modelado_dl if variable.endswith('_estado')
]

vars_categoricas_originales_modelado_dl = [
    variable for variable in vars_categoricas_modelado_dl if not variable.endswith('_estado')
]

modelos_ensemble_dl = list(configuracion['modelos_ensemble_dl'])
pesos_modelos_dl = dict(configuracion['pesos_modelos_dl'])

preprocesador_numerico_final_dl = preprocesamiento_dl['preprocesador_numerico']
encoder_categorico_final_dl = preprocesamiento_dl['encoder_categorico']
indices_inicio_categoricos_final_dl = np.asarray(preprocesamiento_dl['indices_inicio_categoricos'])
vocabulario_total_final_dl = int(preprocesamiento_dl['vocabulario_total'])

rutas_modelos_dl = {
    clave: ruta_modelos_dl / archivo for clave, archivo in metadatos_dl['archivos_modelos'].items()
}

modelos_dl = {
    clave: keras.models.load_model(ruta, compile=False) for clave, ruta in rutas_modelos_dl.items()
}

tipos_entrada = background.dtypes.to_dict()

advertencias = limitaciones['detalle'].dropna().astype(str).tolist()


# =============================================================================
# FUNCIONES DE PREPARACIÓN
# =============================================================================
def preparar_numericas_estados(df, variables, rangos):
    """
    Separa los valores numéricos válidos de los estados asociados
    a códigos especiales.
    """
    valores = pd.DataFrame(index=df.index)
    estados = pd.DataFrame(index=df.index)

    for variable in variables:
        if variable not in rangos:
            raise KeyError(f'No existe rango documental para {variable}.')

        limite_inferior = rangos[variable]['limite_inferior']
        limite_superior = rangos[variable]['limite_superior']

        serie = pd.to_numeric(df[variable], errors='coerce')

        valores_validos = serie.between(limite_inferior, limite_superior)

        valores[variable] = serie.where(valores_validos)

        estado = pd.Series('VALIDO', index=df.index, dtype='object')

        estado.loc[serie.isna()] = 'MISSING'

        codigos_especiales = serie.notna() & ~valores_validos

        estado.loc[codigos_especiales] = 'COD_' + serie.loc[codigos_especiales].astype(str)

        estados[f'{variable}_estado'] = estado

    salida = {
        'valores': valores,
        'estados': estados
    }

    return salida


def preparar_entrada_ml_original(df):
    """
    Reconstruye la entrada ML definitiva desde variables originales.
    """
    preparacion = preparar_numericas_estados(df, vars_numericas_modelado_ml, diccionario_rangos)

    entrada = pd.concat([
        preparacion['valores'], df[vars_categoricas_originales_modelado_ml], preparacion['estados']
    ], axis=1)[columnas_entrada_ml]

    return entrada


def preparar_entrada_dl_original(df):
    """
    Reconstruye las entradas DL definitivas desde variables originales.
    """
    preparacion = preparar_numericas_estados(df, vars_numericas_modelado_dl, diccionario_rangos)

    entrada_numerica = np.asarray(
        preprocesador_numerico_final_dl.transform(preparacion['valores']), dtype=np.float32
    )

    entrada_categorica = pd.concat([
        df[vars_categoricas_originales_modelado_dl], preparacion['estados']
    ], axis=1)[vars_categoricas_modelado_dl]

    entrada_categorica = entrada_categorica.astype('string').fillna('__MISSING__')

    entrada_categorica = (
        encoder_categorico_final_dl.transform(entrada_categorica).astype(np.int32) + 1
    )

    entrada_categorica = (entrada_categorica + indices_inicio_categoricos_final_dl).astype(np.int32)

    entrada = {
        'entrada_numerica': entrada_numerica,
        'entrada_categorica_tokens': entrada_categorica
    }

    return entrada


def comprobar_probabilidades(y_score, n_registros):
    """
    Comprueba dimensión, finitud y rango de las probabilidades.
    """
    y_score = np.asarray(y_score, dtype=float).reshape(-1)

    comprobaciones = {
        'numero_registros': len(y_score) == n_registros,
        'valores_finitos': np.isfinite(y_score).all(),
        'rango_probabilidad': ((y_score >= 0) & (y_score <= 1)).all()
    }

    return comprobaciones


def generar_probabilidades_dl(entradas):
    """
    Genera las probabilidades del ensemble final Deep Learning.
    """
    n_registros = len(entradas['entrada_numerica'])
    probabilidades = {target: np.zeros(n_registros, dtype=float) for target in targets}

    batch_size = 256

    for clave in modelos_ensemble_dl:
        for inicio in range(0, n_registros, batch_size):
            fin = min(inicio + batch_size, n_registros)

            entradas_lote = {nombre: valores[inicio:fin] for nombre, valores in entradas.items()}

            salidas = modelos_dl[clave](entradas_lote, training=False)

            if not isinstance(salidas, dict):
                salidas = dict(zip(
                    modelos_dl[clave].output_names, salidas
                    if isinstance(salidas, list) else [salidas]
                ))

            if set(salidas) != set(targets):
                raise ValueError(f'Las salidas de {clave} no coinciden con las variables objetivo.')

            for target in targets:
                probabilidad = np.asarray(salidas[target]).reshape(-1)

                comprobaciones = comprobar_probabilidades(probabilidad, fin - inicio)

                if not all(comprobaciones.values()):
                    raise ValueError(
                        f'Las probabilidades de {clave} no son correctas para {target}.'
                    )

                probabilidades[target][inicio:fin] += pesos_modelos_dl[clave] * probabilidad

    salida = pd.DataFrame(probabilidades)

    return salida


def predecir_objetivo_original(X, columnas, tipos, target, familia):
    """
    Genera probabilidades reproducibles desde variables originales.
    """
    df = pd.DataFrame(X, columns=columnas).astype(tipos)

    n_registros = len(df)

    df = df.loc[
        df.index.repeat(n_repeticiones_inferencia)
    ].reset_index(drop=True)

    if familia == 'ML':
        entrada = preparar_entrada_ml_original(df)

        probabilidad = modelos_ml[target].predict_proba(
            entrada
        )[:, 1]

    elif familia == 'DL':
        entrada = preparar_entrada_dl_original(df)

        probabilidad = generar_probabilidades_dl(
            entrada
        )[target].to_numpy()

    else:
        raise ValueError(
            f'Familia no reconocida: {familia}'
        )

    probabilidad = np.asarray(
        probabilidad,
        dtype=float
    ).reshape(
        n_registros,
        n_repeticiones_inferencia
    )[:, 0]

    return probabilidad


def convertir_valor_json(valor):
    """
    Convierte un valor de pandas o NumPy a un tipo compatible con JSON.
    """
    if pd.isna(valor):
        salida = None
    elif isinstance(valor, np.generic):
        salida = valor.item()
    else:
        salida = valor

    return salida


def obtener_registro_referencia(questid2):
    """
    Recupera un registro conocido para comprobar la coincidencia de las predicciones.
    """
    columnas = ['QUESTID2'] + vars_predictoras_modelado_dl

    df_referencia = pd.read_parquet(
        ruta_base_premodelado, columns=columnas, filters=[('QUESTID2', '==', questid2)]
    )

    if len(df_referencia) != 1:
        raise ValueError(f'QUESTID2 {questid2} no identifica exactamente un registro.')

    fila = df_referencia.iloc[0]

    registro = {
        variable: convertir_valor_json(fila[variable]) for variable in vars_predictoras_modelado_dl
    }

    salida = {
        'QUESTID2': convertir_valor_json(fila['QUESTID2']),
        'registro': registro
    }

    return salida


# =============================================================================
# PREDICCIÓN
# =============================================================================
def preparar_registro(registro):
    """
    Valida nombres y recupera los tipos originales del registro.
    """
    if not isinstance(registro, dict):
        raise TypeError('El registro debe ser un diccionario.')

    variables_recibidas = set(registro)
    variables_esperadas = set(vars_predictoras_modelado_dl)

    variables_ausentes = [
        variable for variable in vars_predictoras_modelado_dl
        if variable not in variables_recibidas
    ]

    variables_desconocidas = sorted(variables_recibidas - variables_esperadas)

    if variables_ausentes:
        raise ValueError(f'Faltan {len(variables_ausentes)} variables de entrada.')

    if variables_desconocidas:
        raise ValueError(f'Existen {len(variables_desconocidas)} variables desconocidas.')

    df = pd.DataFrame([registro], columns=vars_predictoras_modelado_dl)

    for variable, tipo in tipos_entrada.items():
        df[variable] = df[variable].astype(tipo)

    return df


def generar_predicciones(registro):
    """
    Genera las cuatro predicciones de la solución híbrida final.
    """
    df = preparar_registro(registro)

    df = df.loc[df.index.repeat(n_repeticiones_inferencia)].reset_index(drop=True)

    entrada_ml = preparar_entrada_ml_original(df)
    entrada_dl = preparar_entrada_dl_original(df)

    probabilidades_dl = generar_probabilidades_dl(entrada_dl)

    resultados = []

    for target in targets:
        familia = familias_finales[target]

        if familia == 'ML':
            probabilidad = modelos_ml[target].predict_proba(entrada_ml)[:, 1][0]

        elif familia == 'DL':
            probabilidad = probabilidades_dl.loc[0, target]

        else:
            raise ValueError(f'Familia no reconocida: {familia}')

        probabilidad = float(probabilidad)

        comprobaciones = comprobar_probabilidades([probabilidad], 1)

        if not all(comprobaciones.values()):
            raise ValueError(f'Probabilidad no válida para {target}.')

        umbral = float(umbrales[target])
        clasificacion = int(probabilidad >= umbral)

        resultados.append({
            'variable_objetivo': target,
            'descripcion': titulos_targets[target],
            'probabilidad': probabilidad,
            'umbral': umbral,
            'clasificacion': clasificacion,
            'familia': familia,
            'modelo': modelos_finales[target],
            'variables_principales': None,
            'diferencia_reconstruccion': None,
            'advertencias': advertencias
        })

    return resultados


def generar_explicacion_local(registro, target, n_variables=15):
    """
    Genera una explicación SHAP local desde variables originales.
    """
    import shap

    if target not in targets:
        raise ValueError(f'Variable objetivo desconocida: {target}')

    inicio = perf_counter()

    familia = familias_finales[target]
    columnas = vars_predictoras_modelado_dl
    tipos = tipos_entrada

    df = preparar_registro(registro)

    no_numericas = background[columnas].select_dtypes(exclude=np.number).columns.tolist()

    if no_numericas:
        raise TypeError(
            f'Existen variables originales no numéricas para SHAP: {no_numericas[:10]}.'
        )

    X_background = background[columnas].to_numpy(dtype=float, na_value=np.nan)

    X_registro = df[columnas].to_numpy(dtype=float, na_value=np.nan)

    def funcion(X):
        return predecir_objetivo_original(X, columnas, tipos, target, familia)

    explainer = shap.Explainer(
        funcion, X_background, algorithm='permutation', feature_names=columnas,
        seed=configuracion['semilla']
    )

    n_evaluaciones_shap = 2 * len(columnas) + 1

    explicacion = explainer(
        X_registro, max_evals=n_evaluaciones_shap, batch_size=128, silent=True
    )

    valores_shap = np.asarray(explicacion.values)[0].reshape(-1)

    valor_base = float(np.asarray(explicacion.base_values).reshape(-1)[0])

    probabilidad = float(funcion(X_registro)[0])

    reconstruccion = float(valor_base + valores_shap.sum())

    diferencia_reconstruccion = abs(reconstruccion - probabilidad)

    posiciones = np.argsort(np.abs(valores_shap))[::-1][:n_variables]

    variables_principales = []

    for posicion in posiciones:
        variable = columnas[posicion]
        valor_shap = float(valores_shap[posicion])

        if valor_shap > 0:
            signo = 'Positiva'
        elif valor_shap < 0:
            signo = 'Negativa'
        else:
            signo = 'Nula'

        variables_principales.append({
            'variable': variable,
            'etiqueta_variable': etiquetas_variables.get(variable, variable),
            'valor_observado': convertir_valor_json(df.iloc[0][variable]),
            'valor_shap': valor_shap,
            'signo_contribucion': signo
        })

    salida = {
        'variable_objetivo': target,
        'descripcion': titulos_targets[target],
        'familia': familia,
        'modelo': modelos_finales[target],
        'probabilidad': probabilidad,
        'valor_base': valor_base,
        'reconstruccion': reconstruccion,
        'diferencia_reconstruccion': diferencia_reconstruccion,
        'variables_principales': variables_principales,
        'n_evaluaciones_shap': n_evaluaciones_shap,
        'tiempo_s': perf_counter() - inicio
    }

    return salida


# =============================================================================
# OPERACIONES DEL SERVICIO
# =============================================================================
def comprobar_servicio():
    """
    Resume el estado de los recursos predictivos recuperados.
    """
    pesos_dl_correctos = (
        set(pesos_modelos_dl) == set(modelos_ensemble_dl)
        and all(np.isclose(float(peso), 0.5) for peso in pesos_modelos_dl.values())
    )

    comprobaciones = {
        'entorno': Path(sys.prefix).name == 'TFM_ML',
        'version': configuracion['version'] == 'v1',
        'targets': len(targets) == 4,
        'modelos_ml': len(modelos_ml) == 4,
        'modelos_dl': len(modelos_dl) == 2,
        'variables_entrada': len(vars_predictoras_modelado_dl) == 813,
        'variables_ml': len(vars_predictoras_modelado_ml) == 59,
        'columnas_ml': len(columnas_entrada_ml) == 67,
        'background': background.shape == (20, 813),
        'pesos_dl': pesos_dl_correctos
    }

    salida = {
        'estado': 'OK' if all(comprobaciones.values()) else 'ERROR',
        'operacion': 'health',
        'entorno': Path(sys.prefix).name,
        'version': configuracion['version'],
        'targets': len(targets),
        'modelos_ml': len(modelos_ml),
        'modelos_dl': len(modelos_dl),
        'variables_entrada': len(vars_predictoras_modelado_dl),
        'variables_ml': len(vars_predictoras_modelado_ml),
        'columnas_ml': len(columnas_entrada_ml),
        'background': list(background.shape),
        'comprobaciones': comprobaciones
    }

    return salida


def procesar_solicitud(solicitud):
    """
    Procesa una solicitud JSON.
    """
    if not isinstance(solicitud, dict):
        raise TypeError('La solicitud debe ser un diccionario.')

    operacion = solicitud.get('operacion')

    if operacion == 'health':
        salida = comprobar_servicio()

    elif operacion == 'reference':
        if 'QUESTID2' not in solicitud:
            raise KeyError('La operación reference requiere QUESTID2.')

        referencia = obtener_registro_referencia(solicitud['QUESTID2'])

        salida = {
            'estado': 'OK',
            'operacion': 'reference',
            **referencia
        }

    elif operacion == 'predict':
        if 'registro' not in solicitud:
            raise KeyError('La operación predict requiere un registro.')

        salida = {
            'estado': 'OK',
            'operacion': 'predict',
            'resultados': generar_predicciones(solicitud['registro'])
        }

    elif operacion == 'explain':
        if 'registro' not in solicitud:
            raise KeyError('La operación explain requiere un registro.')

        if 'variable_objetivo' not in solicitud:
            raise KeyError('La operación explain requiere una variable objetivo.')

        salida = {
            'estado': 'OK',
            'operacion': 'explain',
            'explicacion': generar_explicacion_local(
                solicitud['registro'],
                solicitud['variable_objetivo']
            )
        }
    else:
        raise ValueError(f'Operación no reconocida: {operacion}')

    return salida


# =============================================================================
# EJECUCIÓN POR STDIN / STDOUT
# =============================================================================
def main():
    try:
        texto = sys.stdin.read()

        if not texto.strip():
            raise ValueError('No se ha recibido ninguna solicitud.')

        solicitud = json.loads(texto)
        respuesta = procesar_solicitud(solicitud)

    except Exception as exc:
        respuesta = {
            'estado': 'ERROR',
            'tipo_error': type(exc).__name__,
            'mensaje': str(exc)
        }

    print(json.dumps(respuesta, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()