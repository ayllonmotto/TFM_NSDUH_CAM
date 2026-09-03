#!/usr/bin/env python3
# =============================================================================
# TFM NSDUH 2024 — LANZADOR DE DEMOSTRACIÓN VISUAL
# =============================================================================
"""
Abre, sin modificarlo, el Notebook 05 cerrado del proyecto y recupera
el runtime necesario para reproducir funcionalmente la aplicación Flask
utilizada en la demostración del apartado 8.6.

Objetivo:
- no ejecutar Jupyter;
- no modificar el Notebook 05;
- no repetir el Run All;
- no reentrenar modelos;
- mantener TFM_ML en modo de inferencia;
- disponer de la interfaz visual, historial y endpoints de la API;
- mantener el servidor activo hasta pulsar "Salir y cerrar" o Ctrl+C.

La fuente de runtime es el Notebook 05 REAL que exista en:
    ~/BD/TFM_NSDUH/notebooks/

El lanzador resuelve dependencias estáticas entre definiciones del notebook
y evita ejecutar las celdas de pruebas, tablas, guardados y demostraciones
que no son necesarias para servir la aplicación.
"""

from __future__ import annotations

import ast
import builtins
import hashlib
import json
import os
import subprocess
import symtable
import sys
import threading
import time
import traceback
import types
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


# =============================================================================
# CONFIGURACIÓN
# =============================================================================
RUTA_PROYECTO = Path.home() / "BD" / "TFM_NSDUH"
RUTA_NOTEBOOKS = RUTA_PROYECTO / "notebooks"

# Rutas canónicas utilizadas por el Notebook 05.
# Se fuerzan explícitamente para que la demo externa consulte exactamente
# los mismos informes persistidos que la aplicación del apartado 8.6.
RUTA_TABLAS_CANONICA = RUTA_PROYECTO / "results" / "tables" / "05_agentes_productivizacion"
RUTA_FIGURAS_CANONICA = RUTA_PROYECTO / "results" / "figures" / "05_agentes_productivizacion"
RUTA_INFORMES_CANONICA = RUTA_PROYECTO / "results" / "agent_reports"
RUTA_LOGS_CANONICA = RUTA_PROYECTO / "logs"

NOMBRE_NOTEBOOK_PREFERIDO = "05_TFM_Agentes_Productivizacion.ipynb"
PATRON_NOTEBOOK = "05_TFM_Agentes_Productivizacion*.ipynb"

VERSION_LANZADOR = "v3.1-8.6"

HOST = "127.0.0.1"
PUERTO = 5000
URL_ACCESO = f"http://{HOST}:{PUERTO}/acceso"
URL_HEALTH = f"http://{HOST}:{PUERTO}/api/v1/health"

MARCADOR_APP = "# API E INTERFAZ FLASK"

RUTAS_REQUERIDAS = {
    "/acceso",
    "/informes",
    "/informe/<id_informe>",
    "/api/v1/health",
    "/api/v1/esquema",
    "/api/v1/predict",
    "/api/v1/report",
    "/api/v1/access",
    "/api/v1/report-file",
}

FUNCIONES_APP = {
    "acceso_usuario",
    "informes_usuario",
    "informe_guardado",
    "api_health",
    "api_esquema",
    "api_predict",
    "api_report",
    "api_access",
    "api_report_file",
}

# La ruta "/" original depende de una respuesta de demostración construida
# durante el Run All. En este lanzador se reemplaza por una redirección a
# /acceso, evitando recalcular una demostración al arrancar.
FUNCION_RAIZ_ORIGINAL = "interfaz_principal"

NOMBRES_RAIZ = {
    "app",
    "app_api_key",
    *FUNCIONES_APP,
}

BUILTINS = set(dir(builtins))


# =============================================================================
# UTILIDADES
# =============================================================================
def cabecera() -> None:
    print()
    print("=" * 72)
    print("TFM NSDUH 2024 — DEMOSTRACIÓN VISUAL LOCAL")
    print("=" * 72)
    print(f"Proyecto : {RUTA_PROYECTO}")
    print(f"Entorno  : {Path(sys.prefix).name}")
    print(f"Python   : {sys.executable}")
    print(f"Lanzador : {VERSION_LANZADOR}")
    print()


def sha256_archivo(ruta: Path) -> str:
    h = hashlib.sha256()
    with ruta.open("rb") as f:
        for bloque in iter(lambda: f.read(1024 * 1024), b""):
            h.update(bloque)
    return h.hexdigest()


def localizar_notebook() -> Path:
    preferido = RUTA_NOTEBOOKS / NOMBRE_NOTEBOOK_PREFERIDO
    if preferido.exists():
        return preferido

    candidatos = sorted(
        RUTA_NOTEBOOKS.glob(PATRON_NOTEBOOK),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidatos:
        raise FileNotFoundError(
            "No se ha localizado el Notebook 05 en "
            f"{RUTA_NOTEBOOKS}.\n"
            f"Se esperaba {NOMBRE_NOTEBOOK_PREFERIDO} o un archivo compatible."
        )

    print("[AVISO] No existe el nombre preferido; se utilizará el Notebook 05")
    print("        compatible con fecha de modificación más reciente:")
    print(f"        {candidatos[0]}")
    return candidatos[0]


def cargar_celdas_codigo(ruta_notebook: Path) -> list[tuple[int, str]]:
    with ruta_notebook.open("r", encoding="utf-8") as f:
        notebook = json.load(f)

    celdas = []
    for indice, celda in enumerate(notebook.get("cells", [])):
        if celda.get("cell_type") != "code":
            continue
        fuente = celda.get("source", "")
        if isinstance(fuente, list):
            fuente = "".join(fuente)
        celdas.append((indice, fuente))

    if not celdas:
        raise RuntimeError("El Notebook 05 no contiene celdas de código.")

    return celdas


def nombre_base_objetivo(target: ast.AST) -> str | None:
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, (ast.Attribute, ast.Subscript)):
        valor = target.value
        while isinstance(valor, (ast.Attribute, ast.Subscript)):
            valor = valor.value
        if isinstance(valor, ast.Name):
            return valor.id
    if isinstance(target, (ast.Tuple, ast.List)):
        for elemento in target.elts:
            nombre = nombre_base_objetivo(elemento)
            if nombre:
                return nombre
    return None


def nombres_definidos(node: ast.AST) -> set[str]:
    nombres: set[str] = set()

    if isinstance(node, ast.Import):
        for alias in node.names:
            nombres.add(alias.asname or alias.name.split(".")[0])
        return nombres

    if isinstance(node, ast.ImportFrom):
        for alias in node.names:
            if alias.name != "*":
                nombres.add(alias.asname or alias.name)
        return nombres

    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {node.name}

    # Para asignaciones/estructuras de control interesa conocer los nombres
    # realmente escritos en el ámbito de módulo.
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
            nombres.add(sub.id)

    return nombres


def bases_mutadas(node: ast.AST) -> set[str]:
    bases: set[str] = set()

    for sub in ast.walk(node):
        if isinstance(sub, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets: list[ast.AST] = []
            if isinstance(sub, ast.Assign):
                targets = list(sub.targets)
            else:
                targets = [sub.target]
            for target in targets:
                if isinstance(target, (ast.Attribute, ast.Subscript)):
                    nombre = nombre_base_objetivo(target)
                    if nombre:
                        bases.add(nombre)

        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute):
            valor = sub.func.value
            while isinstance(valor, (ast.Attribute, ast.Subscript)):
                valor = valor.value
            if isinstance(valor, ast.Name):
                bases.add(valor.id)

    return bases


def _referencias_globales_tabla(tabla: symtable.SymbolTable) -> set[str]:
    refs: set[str] = set()
    for simbolo in tabla.get_symbols():
        if simbolo.is_referenced() and simbolo.is_global():
            refs.add(simbolo.get_name())
    for hija in tabla.get_children():
        refs |= _referencias_globales_tabla(hija)
    return refs


def referencias_node(node: ast.AST) -> set[str]:
    # Para funciones y clases, symtable evita confundir variables locales
    # con globals del notebook.
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        try:
            codigo = ast.unparse(node)
            tabla = symtable.symtable(codigo, "<tfm_runtime>", "exec")
            refs = _referencias_globales_tabla(tabla)
        except Exception:
            refs = {
                sub.id
                for sub in ast.walk(node)
                if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load)
            }
    else:
        refs = {
            sub.id
            for sub in ast.walk(node)
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load)
        }

    return {r for r in refs if r not in BUILTINS}



def contiene_await_nivel_superior(node: ast.AST) -> bool:
    """
    Devuelve True cuando el bloque requiere top-level await.

    Jupyter/IPython admite ``await`` en una celda, pero un archivo .py normal
    no puede compilar expresiones como ``x = await ...`` fuera de una función
    async. Las funciones ``async def`` sí son válidas y no se consideran aquí.
    """
    def visitar(actual: ast.AST) -> bool:
        if isinstance(actual, ast.Await):
            return True

        # No descender en ámbitos donde await sí es sintácticamente válido.
        if isinstance(
            actual,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef),
        ):
            return False

        return any(visitar(hijo) for hijo in ast.iter_child_nodes(actual))

    return visitar(node)


METODOS_CONFIGURACION = {
    "add_node",
    "add_edge",
    "add_conditional_edges",
    "append",
    "extend",
    "update",
    "mkdir",
}


def bases_configuradas(node: ast.AST) -> set[str]:
    """
    Identifica objetos que se configuran de forma incremental.

    Se limita deliberadamente a operaciones de construcción/configuración.
    No considera llamadas como ``tool.invoke()``, ``ainvoke()``, ``predict()``
    o similares, porque son ejecuciones/pruebas y no forman parte del arranque
    de la demo.
    """
    bases: set[str] = set()

    for sub in ast.walk(node):
        # Asignación de atributos: app.secret_key = ..., Settings.embed_model = ...
        if isinstance(sub, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = list(sub.targets) if isinstance(sub, ast.Assign) else [sub.target]
            for target in targets:
                if isinstance(target, ast.Attribute):
                    valor = target.value
                    while isinstance(valor, (ast.Attribute, ast.Subscript)):
                        valor = valor.value
                    if isinstance(valor, ast.Name):
                        bases.add(valor.id)

        # Construcción incremental: builder.add_node(...), lista.append(...), etc.
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute):
            if sub.func.attr not in METODOS_CONFIGURACION:
                continue

            valor = sub.func.value
            while isinstance(valor, (ast.Attribute, ast.Subscript)):
                valor = valor.value

            if isinstance(valor, ast.Name):
                bases.add(valor.id)

    return bases


def llamada_simple(node: ast.AST, nombre: str) -> bool:
    if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
        return False
    func = node.value.func
    return isinstance(func, ast.Name) and func.id == nombre


@dataclass
class RegistroNodo:
    orden: int
    celda: int
    nodo: int
    ast_node: ast.AST
    definidos: set[str]
    referencias: set[str]
    mutados: set[str]


def preparar_registros(celdas: list[tuple[int, str]]) -> tuple[list[RegistroNodo], int]:
    registros: list[RegistroNodo] = []
    orden = 0
    celda_app = None

    for indice_celda, fuente in celdas:
        if celda_app is None and MARCADOR_APP in fuente:
            celda_app = indice_celda

        # Solo necesitamos definiciones hasta la celda Flask incluida.
        if celda_app is not None and indice_celda > celda_app:
            break

        try:
            arbol = ast.parse(fuente, filename=f"<notebook_celda_{indice_celda}>")
        except SyntaxError as exc:
            raise RuntimeError(
                f"No se ha podido analizar la celda {indice_celda}: {exc}"
            ) from exc

        for indice_nodo, node in enumerate(arbol.body):
            registros.append(
                RegistroNodo(
                    orden=orden,
                    celda=indice_celda,
                    nodo=indice_nodo,
                    ast_node=node,
                    definidos=nombres_definidos(node),
                    referencias=referencias_node(node),
                    mutados=bases_mutadas(node),
                )
            )
            orden += 1

    if celda_app is None:
        raise RuntimeError(
            "No se ha localizado la celda '# API E INTERFAZ FLASK' "
            "en el Notebook 05."
        )

    return registros, celda_app


def seleccionar_runtime(registros: list[RegistroNodo]) -> list[RegistroNodo]:
    """
    Reconstruye únicamente el runtime necesario para la aplicación Flask.

    Criterios:
    - respeta la definición temporal correcta de cada nombre;
    - excluye pruebas y ejecuciones de validación;
    - excluye celdas con top-level await (MCP de validación);
    - conserva únicamente mutaciones de configuración necesarias;
    - no ejecuta SHAP, predicciones de demostración ni MCP al arrancar.
    """

    definiciones_por_nombre: dict[str, list[RegistroNodo]] = {}
    for reg in registros:
        for nombre in reg.definidos:
            definiciones_por_nombre.setdefault(nombre, []).append(reg)

    def resolver_definicion(
        nombre: str,
        antes_de: int | None = None,
    ) -> RegistroNodo | None:
        candidatos = definiciones_por_nombre.get(nombre, [])
        if not candidatos:
            return None

        if antes_de is None:
            candidatos_validos = candidatos
        else:
            candidatos_validos = [
                reg for reg in candidatos if reg.orden < antes_de
            ]

        # Si una definición concreta contiene top-level await, se omite.
        # El MCP del notebook es una validación interoperable, no una
        # dependencia del flujo principal de la aplicación.
        candidatos_validos = [
            reg for reg in candidatos_validos
            if not contiene_await_nivel_superior(reg.ast_node)
        ]

        return candidatos_validos[-1] if candidatos_validos else None

    seleccion: set[int] = set()

    # Importaciones: son definiciones, no ejecutan las pruebas del notebook.
    for reg in registros:
        if isinstance(reg.ast_node, (ast.Import, ast.ImportFrom)):
            seleccion.add(reg.orden)

    raiz_reg = resolver_definicion(FUNCION_RAIZ_ORIGINAL)

    pendientes: list[tuple[str, int | None]] = [
        (nombre, None) for nombre in NOMBRES_RAIZ
    ]
    visitados: set[tuple[str, int | None]] = set()

    def incorporar_dependencias(reg: RegistroNodo) -> None:
        for referencia in reg.referencias:
            pendientes.append((referencia, reg.orden))

    # ------------------------------------------------------------------
    # Dependencias directas/recursivas
    # ------------------------------------------------------------------
    while pendientes:
        nombre, antes_de = pendientes.pop()
        clave = (nombre, antes_de)

        if clave in visitados:
            continue
        visitados.add(clave)

        reg = resolver_definicion(nombre, antes_de)
        if reg is None:
            continue

        if raiz_reg is not None and reg.orden == raiz_reg.orden:
            continue

        if reg.orden not in seleccion:
            seleccion.add(reg.orden)
            incorporar_dependencias(reg)

    # ------------------------------------------------------------------
    # Mutaciones de configuración
    #
    # Solo se incorporan operaciones de construcción sobre objetos
    # definidos por el runtime: LangGraph, listas documentales, Flask, etc.
    #
    # NO se incorporan llamadas de ejecución como:
    #   tool.invoke(), ainvoke(), predict(), explicar..., etc.
    # ------------------------------------------------------------------
    cambio = True
    while cambio:
        cambio = False

        nombres_objeto: set[str] = set()
        for reg in registros:
            if reg.orden not in seleccion:
                continue
            if isinstance(reg.ast_node, (ast.Import, ast.ImportFrom)):
                continue
            nombres_objeto |= reg.definidos

        # Settings es un singleton importado que LlamaIndex configura
        # mediante asignación de atributos.
        bases_admitidas = nombres_objeto | {"Settings"}

        nuevos: list[RegistroNodo] = []

        for reg in registros:
            if reg.orden in seleccion:
                continue
            if raiz_reg is not None and reg.orden == raiz_reg.orden:
                continue
            if contiene_await_nivel_superior(reg.ast_node):
                continue

            bases = bases_configuradas(reg.ast_node)
            mutacion_necesaria = bool(bases & bases_admitidas)
            carga_env = llamada_simple(reg.ast_node, "load_dotenv")

            if mutacion_necesaria or carga_env:
                nuevos.append(reg)

        if nuevos:
            for reg in nuevos:
                seleccion.add(reg.orden)
                incorporar_dependencias(reg)
            cambio = True

        while pendientes:
            nombre, antes_de = pendientes.pop()
            clave = (nombre, antes_de)

            if clave in visitados:
                continue
            visitados.add(clave)

            reg = resolver_definicion(nombre, antes_de)
            if reg is None:
                continue
            if raiz_reg is not None and reg.orden == raiz_reg.orden:
                continue

            if reg.orden not in seleccion:
                seleccion.add(reg.orden)
                incorporar_dependencias(reg)
                cambio = True

    # La ruta "/" original usa una respuesta ya precalculada durante el
    # Run All. La demo externa redirige "/" a /acceso.
    nombres_excluir = {
        "factores_interfaz",
        "detalle_interfaz",
        "respuesta_demo_aplicacion",
        "figura_predicciones_html",
        "informe_validado",
        # Elementos exclusivos de la validación MCP.
        "cliente_mcp",
        "tools_mcp",
        "nombres_tools_mcp",
        "tool_predictiva_mcp",
        "resultado_mcp",
        "respuesta_tool_mcp",
        "resultados_tool_mcp",
        "predicciones_tool_mcp",
    }

    seleccion_final: list[RegistroNodo] = []

    for reg in registros:
        if reg.orden not in seleccion:
            continue
        if contiene_await_nivel_superior(reg.ast_node):
            continue
        if reg.definidos & nombres_excluir:
            continue
        if (
            isinstance(reg.ast_node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and reg.ast_node.name == FUNCION_RAIZ_ORIGINAL
        ):
            continue

        seleccion_final.append(reg)

    return seleccion_final


def ejecutar_runtime(
    registros: list[RegistroNodo],
    ruta_notebook: Path,
) -> dict:
    """
    Ejecuta el runtime seleccionado imitando la semántica normal del notebook.

    Dos detalles son críticos:
    1. el módulo se registra realmente en ``sys.modules`` para que Pydantic
       pueda resolver namespaces y referencias de tipos;
    2. ``compile(..., dont_inherit=True)`` evita heredar
       ``from __future__ import annotations`` del lanzador. El Notebook 05 no
       usa ese future import, por lo que sus anotaciones (por ejemplo
       ``Literal[0, 1]``) deben evaluarse igual que cuando se ejecutan en
       Jupyter.
    """
    nombre_modulo = "tfm_demo_runtime"

    modulo = types.ModuleType(nombre_modulo)
    modulo.__file__ = str(ruta_notebook)
    modulo.__package__ = None

    # Pydantic consulta el namespace del módulo al reconstruir modelos.
    sys.modules[nombre_modulo] = modulo
    ns = modulo.__dict__

    # Trabajar desde la raíz real del proyecto.
    os.chdir(RUTA_PROYECTO)

    for reg in registros:
        modulo_ast = ast.Module(body=[reg.ast_node], type_ignores=[])
        ast.fix_missing_locations(modulo_ast)

        try:
            codigo = compile(
                modulo_ast,
                filename=f"{ruta_notebook.name}:celda_{reg.celda}",
                mode="exec",
                dont_inherit=True,
            )
            exec(codigo, ns)
        except Exception as exc:
            nombre = ", ".join(sorted(reg.definidos)) or type(reg.ast_node).__name__
            raise RuntimeError(
                f"Error inicializando '{nombre}' desde la celda {reg.celda}: {exc}"
            ) from exc

    return ns


def reconstruir_modelos_pydantic(ns: dict) -> None:
    """
    Verificación defensiva posterior.

    El Notebook 05 funciona en Jupyter sin reconstrucciones manuales porque
    sus tipos están disponibles en el namespace interactivo. En el lanzador
    externo se fuerza la reconstrucción de todos los BaseModel recuperados
    usando exactamente el namespace del runtime.
    """
    try:
        from pydantic import BaseModel
    except Exception as exc:
        raise RuntimeError(f"No se ha podido importar Pydantic: {exc}") from exc

    modelos = []
    for nombre, objeto in list(ns.items()):
        if (
            isinstance(objeto, type)
            and issubclass(objeto, BaseModel)
            and objeto is not BaseModel
            and getattr(objeto, "__module__", None) == "tfm_demo_runtime"
        ):
            modelos.append((nombre, objeto))

    errores = []
    for nombre, modelo in modelos:
        try:
            modelo.model_rebuild(
                force=True,
                _types_namespace=ns,
            )
        except Exception as exc:
            errores.append(f"{nombre}: {exc}")

    if errores:
        raise RuntimeError(
            "No se han podido reconstruir los contratos Pydantic:\n  - "
            + "\n  - ".join(errores)
        )

    print(f"Contratos Pydantic     : {len(modelos)} reconstruidos")


def fijar_rutas_canonicas_runtime(ns: dict) -> None:
    """
    Impone las mismas rutas persistentes que usa el Notebook 05.

    Esto es importante porque las funciones Flask recuperadas del notebook
    resuelven sus variables globales en tiempo de petición. Al fijar aquí
    ``ruta_informes`` se garantiza que /informes y /informe/<id> consultan
    exactamente ~/BD/TFM_NSDUH/results/agent_reports.
    """
    ns["ruta_proyecto"] = RUTA_PROYECTO
    ns["ruta_tablas"] = RUTA_TABLAS_CANONICA
    ns["ruta_figuras"] = RUTA_FIGURAS_CANONICA
    ns["ruta_informes"] = RUTA_INFORMES_CANONICA
    ns["ruta_logs"] = RUTA_LOGS_CANONICA

    for ruta in (
        RUTA_TABLAS_CANONICA,
        RUTA_FIGURAS_CANONICA,
        RUTA_INFORMES_CANONICA,
        RUTA_LOGS_CANONICA,
    ):
        ruta.mkdir(parents=True, exist_ok=True)


def comprobar_historial_persistido(ns: dict) -> list[dict]:
    """
    Comprueba antes de arrancar Flask que la demo ve el mismo historial
    persistido que el Notebook 05.
    """
    archivos = sorted(RUTA_INFORMES_CANONICA.glob("informe_*.json"))

    print("Persistencia:")
    print(f"  Ruta de informes      : {RUTA_INFORMES_CANONICA}")
    print(f"  JSON encontrados      : {len(archivos)}")

    cargar_historial = ns.get("cargar_historial_informes")
    if not callable(cargar_historial):
        raise RuntimeError(
            "No se ha recuperado cargar_historial_informes(), necesaria "
            "para reproducir el historial de la aplicación."
        )

    historial = cargar_historial()

    print(f"  Informes compatibles  : {len(historial)}")

    ids = [str(item.get("id_informe", "")) for item in historial]
    if ids:
        vista = ", ".join(ids[:10])
        if len(ids) > 10:
            vista += ", ..."
        print(f"  Identificadores       : {vista}")
    else:
        print("  Identificadores       : ninguno")

    demos_esperadas = {"demo_01", "demo_02", "demo_03"}
    demos_presentes = demos_esperadas.intersection(ids)
    if demos_presentes:
        print(
            "  Informes demo         : "
            + ", ".join(sorted(demos_presentes))
        )

    print()
    return historial


def comprobar_runtime(ns: dict) -> None:
    if "app" not in ns:
        raise RuntimeError("No se ha recuperado el objeto Flask 'app'.")

    app = ns["app"]

    # Sustitución segura de "/" para no requerir una respuesta precalculada.
    if not any(rule.rule == "/" for rule in app.url_map.iter_rules()):
        from flask import redirect, url_for

        @app.get("/")
        def inicio_demo_local():
            return redirect(url_for("acceso_usuario"))

    rutas = {rule.rule for rule in app.url_map.iter_rules()}
    faltan = sorted(RUTAS_REQUERIDAS - rutas)

    if faltan:
        raise RuntimeError(
            "La aplicación recuperada no contiene todas las rutas esperadas:\n"
            + "\n".join(f"  - {ruta}" for ruta in faltan)
        )

    # Asegurar las carpetas de persistencia si están presentes en el runtime.
    for nombre in ("ruta_informes", "ruta_logs"):
        ruta = ns.get(nombre)
        if isinstance(ruta, Path):
            ruta.mkdir(parents=True, exist_ok=True)


def abrir_navegador(url: str) -> None:
    # Desde WSL se prioriza el navegador de Windows.
    try:
        subprocess.Popen(
            ["cmd.exe", "/c", "start", "", url],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return
    except Exception:
        pass

    try:
        import webbrowser
        webbrowser.open(url)
    except Exception:
        print(f"[AVISO] Abra manualmente: {url}")


def esperar_health(timeout: float = 30.0) -> dict | None:
    limite = time.time() + timeout
    while time.time() < limite:
        try:
            with urllib.request.urlopen(URL_HEALTH, timeout=2) as respuesta:
                if respuesta.status == 200:
                    return json.loads(respuesta.read().decode("utf-8"))
        except Exception:
            time.sleep(0.4)
    return None


# =============================================================================
# ARRANQUE
# =============================================================================
def main() -> int:
    cabecera()

    if Path(sys.prefix).name != "TFM_Agentes":
        print("[ERROR] El lanzador debe ejecutarse en el entorno TFM_Agentes.")
        print(f"        Entorno detectado: {Path(sys.prefix).name}")
        return 2

    if not RUTA_PROYECTO.exists():
        print(f"[ERROR] No existe el proyecto: {RUTA_PROYECTO}")
        return 2

    try:
        ruta_notebook = localizar_notebook()
        print(f"Notebook : {ruta_notebook}")
        print(f"SHA-256  : {sha256_archivo(ruta_notebook)}")
        print()
        print("Preparando runtime de la aplicación...")
        print("No se modifica ni se guarda el Notebook 05.")

        celdas = cargar_celdas_codigo(ruta_notebook)
        registros, celda_app = preparar_registros(celdas)
        seleccion = seleccionar_runtime(registros)

        celdas_runtime = sorted({reg.celda for reg in seleccion})
        await_runtime = [
            reg for reg in seleccion if contiene_await_nivel_superior(reg.ast_node)
        ]

        print(f"Celda Flask localizada : {celda_app}")
        print(f"Bloques runtime        : {len(seleccion)}")
        print(f"Celdas runtime         : {len(celdas_runtime)}")
        print(f"Top-level await        : {len(await_runtime)}")

        if await_runtime:
            raise RuntimeError(
                "El selector ha incluido código con top-level await; "
                "se aborta antes de ejecutar el runtime."
            )

        print()

        ns = ejecutar_runtime(seleccion, ruta_notebook)

        # Igualar el namespace interactivo de Jupyter para los contratos
        # Pydantic antes de validar informes persistidos.
        reconstruir_modelos_pydantic(ns)

        # Reproducir exactamente la persistencia utilizada por el Notebook 05.
        fijar_rutas_canonicas_runtime(ns)
        comprobar_runtime(ns)
        historial_inicial = comprobar_historial_persistido(ns)

        from werkzeug.serving import make_server

        app = ns["app"]

        # Igual que la celda 8.6: servidor Werkzeug local y threaded=True.
        servidor = make_server(HOST, PUERTO, app, threaded=True)

        def detener_servidor_demo(servidor_obj):
            try:
                servidor_obj.shutdown()
            except Exception:
                pass

        # Las funciones de /acceso fueron compiladas con este mismo namespace.
        # Por ello globals() dentro de la ruta verá estas referencias.
        ns["detener_servidor_demo"] = detener_servidor_demo
        ns["servidor_demo"] = servidor
        ns["temporizador_servidor_demo"] = None

        hilo = threading.Thread(
            target=servidor.serve_forever,
            name="TFM_Demo_Flask",
            daemon=False,
        )
        hilo.start()

        health = esperar_health()
        if health is None:
            servidor.shutdown()
            hilo.join(timeout=5)
            raise RuntimeError(
                "El servidor se inició, pero /api/v1/health no respondió a tiempo."
            )

        print("=" * 72)
        print("DEMO LISTA")
        print("=" * 72)
        print(f"Interfaz : {URL_ACCESO}")
        print(f"Historial: http://{HOST}:{PUERTO}/informes")
        print(f"Informes : {len(historial_inicial)} compatibles al iniciar")
        print("Acceso   : CAM")
        print(f"Health   : {URL_HEALTH}")
        print()
        print(f"Versión predictiva : {health.get('version_predictiva', 'No disponible')}")
        print(f"Modelo generativo  : {health.get('modelo_generativo', 'No disponible')}")
        print(f"Versión del prompt : {health.get('version_prompt', 'No disponible')}")
        print()
        print("La demo permanecerá activa hasta:")
        print("  - pulsar 'Salir y cerrar' en la interfaz, o")
        print("  - pulsar Ctrl+C en esta ventana.")
        print("=" * 72)
        print()

        abrir_navegador(URL_ACCESO)

        try:
            while hilo.is_alive():
                hilo.join(timeout=0.5)
        except KeyboardInterrupt:
            print("\nCerrando servidor local...")
            servidor.shutdown()
            hilo.join(timeout=10)

        print("Servidor detenido. Demostración finalizada.")
        return 0

    except OSError as exc:
        # Caso frecuente: puerto 5000 ocupado.
        if getattr(exc, "errno", None) in {98, 10048}:
            print("[ERROR] El puerto 5000 ya está ocupado.")
            print("        Cierre la instancia anterior de la demo y vuelva a intentarlo.")
        else:
            print(f"[ERROR] {exc}")
        return 1

    except Exception as exc:
        print()
        print("[ERROR] No ha sido posible iniciar la demostración.")
        print(str(exc))
        print()
        print("Detalle técnico:")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
