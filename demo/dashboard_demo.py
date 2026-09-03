#!/usr/bin/env python3
# =============================================================================
# TFM NSDUH 2024 — DASHBOARD DEMO LOCAL
# =============================================================================
"""
Capa de producto sobre demo/demo_tfm.py.

v3:
- corrige referencias globales definidas después de funciones del Notebook 05;
- mantiene aplicacion_flujo como raíz del runtime real;
- añade historial propio del dashboard leyendo results/agent_reports;
- separa "volver" de "cerrar demo";
- no modifica Notebook 05, modelos, umbrales, SHAP, RAG ni guardrails.
"""

from __future__ import annotations

import ast
import json
import re
import threading
import secrets
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import demo_tfm as base


RUTA_DASHBOARD = Path(__file__).resolve().with_name("dashboard_demo.html")
RUTA_ASSETS = Path(__file__).resolve().with_name("dashboard_assets")
VERSION_DASHBOARD = "dashboard-demo-final"
MARCADOR_DASHBOARD = '<meta name="dashboard-version" content="dashboard-demo-final">'

_seleccionar_runtime_original = base.seleccionar_runtime
_comprobar_runtime_original = base.comprobar_runtime

NOMBRES_EXCLUIR = {
    "factores_interfaz",
    "detalle_interfaz",
    "respuesta_demo_aplicacion",
    "figura_predicciones_html",
    "informe_validado",
    "cliente_mcp",
    "tools_mcp",
    "nombres_tools_mcp",
    "tool_predictiva_mcp",
    "resultado_mcp",
    "respuesta_tool_mcp",
    "resultados_tool_mcp",
    "predicciones_tool_mcp",
}


def _resolver_definicion(
    nombre: str,
    definiciones_por_nombre: dict[str, list],
    antes_de: int | None = None,
):
    candidatos = definiciones_por_nombre.get(nombre, [])

    if antes_de is not None:
        candidatos = [reg for reg in candidatos if reg.orden < antes_de]

    candidatos = [
        reg for reg in candidatos
        if not base.contiene_await_nivel_superior(reg.ast_node)
        and not (reg.definidos & NOMBRES_EXCLUIR)
        and not (
            isinstance(reg.ast_node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and reg.ast_node.name == base.FUNCION_RAIZ_ORIGINAL
        )
    ]

    return candidatos[-1] if candidatos else None


def seleccionar_runtime_robusto(registros):
    """
    Amplía el selector original sin ejecutar el notebook completo.

    El selector v3.1-8.6 resuelve una referencia de una función usando
    exclusivamente definiciones anteriores a esa función. Eso es correcto
    para expresiones ejecutadas inmediatamente, pero no para globals de una
    función Python: éstos se resuelven cuando la función se EJECUTA y pueden
    haberse definido más adelante.

    Ejemplo real detectado:
        nodo_validar_entrada() -> EntradaPredictiva

    La función existía, pero EntradaPredictiva no había entrado en el runtime.
    Esta pasada adicional repara de forma general ese patrón.
    """
    seleccion = list(_seleccionar_runtime_original(registros))
    seleccion_por_orden = {reg.orden: reg for reg in seleccion}

    definiciones_por_nombre: dict[str, list] = {}
    for reg in registros:
        for nombre in reg.definidos:
            definiciones_por_nombre.setdefault(nombre, []).append(reg)

    visitados: set[tuple[str, int | None]] = set()

    def incorporar_referencias(reg):
        # Los globals de una función se resuelven al invocarla, no al definirla.
        # Por ello se permite recuperar una definición posterior.
        limite = None if isinstance(
            reg.ast_node, (ast.FunctionDef, ast.AsyncFunctionDef)
        ) else reg.orden

        for referencia in reg.referencias:
            pendientes.append((referencia, limite))

    pendientes: list[tuple[str, int | None]] = []

    for reg in list(seleccion_por_orden.values()):
        incorporar_referencias(reg)

    def cerrar_dependencias():
        cambio_local = False

        while pendientes:
            nombre, antes_de = pendientes.pop()
            clave = (nombre, antes_de)

            if clave in visitados:
                continue
            visitados.add(clave)

            reg = _resolver_definicion(
                nombre,
                definiciones_por_nombre,
                antes_de=antes_de,
            )
            if reg is None:
                continue

            if reg.orden not in seleccion_por_orden:
                seleccion_por_orden[reg.orden] = reg
                incorporar_referencias(reg)
                cambio_local = True

        return cambio_local

    cerrar_dependencias()

    # Repetir la lógica de configuración incremental para cualquier objeto
    # nuevo incorporado en la pasada anterior.
    cambio = True
    while cambio:
        cambio = False

        nombres_objeto: set[str] = set()
        for reg in seleccion_por_orden.values():
            if isinstance(reg.ast_node, (ast.Import, ast.ImportFrom)):
                continue
            nombres_objeto |= reg.definidos

        bases_admitidas = nombres_objeto | {"Settings"}

        for reg in registros:
            if reg.orden in seleccion_por_orden:
                continue
            if base.contiene_await_nivel_superior(reg.ast_node):
                continue
            if reg.definidos & NOMBRES_EXCLUIR:
                continue

            bases = base.bases_configuradas(reg.ast_node)
            mutacion_necesaria = bool(bases & bases_admitidas)
            carga_env = base.llamada_simple(reg.ast_node, "load_dotenv")

            if mutacion_necesaria or carga_env:
                seleccion_por_orden[reg.orden] = reg
                incorporar_referencias(reg)
                cambio = True

        if cerrar_dependencias():
            cambio = True

    salida = [
        reg for _, reg in sorted(seleccion_por_orden.items())
        if not base.contiene_await_nivel_superior(reg.ast_node)
        and not (reg.definidos & NOMBRES_EXCLUIR)
    ]

    return salida


def _es_peticion_salida(request) -> bool:
    textos = [
        *(str(k).strip().lower() for k in request.form.keys()),
        *(str(v).strip().lower() for v in request.form.values()),
    ]
    return any(
        texto in {
            "salir",
            "cerrar",
            "salir y cerrar",
            "cerrar demo",
            "cerrar servidor",
        }
        for texto in textos
    )


def _cargar_archivo_informe(ruta: Path) -> dict | None:
    try:
        with ruta.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _listar_informes_directos() -> list[dict]:
    items = []

    for ruta in sorted(
        base.RUTA_INFORMES_CANONICA.glob("informe_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ):
        data = _cargar_archivo_informe(ruta)
        if not data:
            continue

        respuesta = data.get("respuesta", {})
        predicciones = respuesta.get("predicciones", [])
        sobre_umbral = sum(
            1 for p in predicciones
            if int(p.get("clasificacion", 0)) == 1
        )

        items.append({
            "id_informe": str(data.get("id_informe", ruta.stem.removeprefix("informe_"))),
            "fecha": str(data.get("fecha", "")),
            "archivo_origen": str(data.get("archivo_origen", "")),
            "sobre_umbral": sobre_umbral,
            "archivo": ruta.name,
        })

    return items


def _buscar_informe_por_id(id_informe: str) -> dict | None:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", id_informe):
        return None

    # Primero intenta el convenio canónico.
    candidatos = [
        base.RUTA_INFORMES_CANONICA / f"informe_{id_informe}.json",
        base.RUTA_INFORMES_CANONICA / f"{id_informe}.json",
    ]

    for ruta in candidatos:
        if ruta.is_file():
            data = _cargar_archivo_informe(ruta)
            if data:
                return data

    # Si el nombre físico no coincide, buscar por id_informe dentro del JSON.
    for ruta in base.RUTA_INFORMES_CANONICA.glob("informe_*.json"):
        data = _cargar_archivo_informe(ruta)
        if data and str(data.get("id_informe", "")) == id_informe:
            return data

    return None


def comprobar_historial_directo(ns: dict) -> list[dict]:
    """
    La portada del dashboard no necesita ejecutar helpers del Notebook 05 para
    listar archivos ya guardados. Lee directamente el contrato JSON final.
    """
    items = _listar_informes_directos()

    print("Informes disponibles:")
    print(f"  Ruta                 : {base.RUTA_INFORMES_CANONICA}")
    print(f"  JSON compatibles     : {len(items)}")

    ids = [x["id_informe"] for x in items]
    if ids:
        vista = ", ".join(ids[:10])
        if len(ids) > 10:
            vista += ", ..."
        print(f"  Identificadores      : {vista}")
    else:
        print("  Identificadores      : ninguno")

    print()
    return items



# =============================================================================
# DASHBOARD INMEDIATO + MOTOR EN SEGUNDO PLANO
# =============================================================================

@dataclass
class EstadoRuntime:
    estado: str = "pendiente"          # pendiente | inicializando | listo | error
    etapa: str = "Pendiente de inicio"
    detalle: str = ""
    progreso: int = 0
    app_real: object | None = None
    ns: dict | None = None
    historial_inicial: list[dict] = field(default_factory=list)
    error: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def actualizar(
        self,
        *,
        estado: str | None = None,
        etapa: str | None = None,
        detalle: str | None = None,
        progreso: int | None = None,
        error: str | None = None,
    ) -> None:
        with self.lock:
            if estado is not None:
                self.estado = estado
            if etapa is not None:
                self.etapa = etapa
            if detalle is not None:
                self.detalle = detalle
            if progreso is not None:
                self.progreso = progreso
            if error is not None:
                self.error = error

    def serializar(self) -> dict:
        with self.lock:
            return {
                "estado": self.estado,
                "etapa": self.etapa,
                "detalle": self.detalle,
                "progreso": self.progreso,
                "listo": self.estado == "listo",
                "error": self.error,
            }


estado_runtime = EstadoRuntime()
servidor_dashboard = None


def cabecera_dashboard_demo() -> None:
    print()
    print("=" * 72)
    print("TFM NSDUH 2024 — DASHBOARD DEMO LOCAL")
    print("=" * 72)
    print(f"Proyecto : {base.RUTA_PROYECTO}")
    print(f"Entorno  : {Path(__import__('sys').prefix).name}")
    print(f"Python   : {__import__('sys').executable}")
    print(f"Lanzador : {base.VERSION_LANZADOR}+{VERSION_DASHBOARD}")
    print()


def _validar_runtime_funcional(ns: dict) -> None:
    """
    Conserva las comprobaciones de la versión estable, sin registrar
    las rutas del dashboard dentro de la app recuperada.
    """
    _comprobar_runtime_original(ns)

    nombres_necesarios = {
        "EntradaPredictiva",
        "aplicacion_flujo",
        "ejecutar_flujo_aplicacion",
        "ejecutar_informe_aplicacion",
    }

    faltantes = sorted(
        nombre for nombre in nombres_necesarios
        if nombre not in ns
    )
    if faltantes:
        raise RuntimeError(
            "Runtime incompleto para la ejecución real. Faltan: "
            + ", ".join(faltantes)
        )

    if not hasattr(ns["aplicacion_flujo"], "invoke"):
        raise RuntimeError(
            "`aplicacion_flujo` existe pero no expone invoke()."
        )

    print("[OK] EntradaPredictiva disponible.")
    print("[OK] aplicacion_flujo disponible.")
    print("[OK] Historial del dashboard disponible.")


def _adaptar_salida_aplicacion(ns: dict) -> None:
    """
    La app real mantiene su comportamiento. Solo se cambia la salida para que
    vuelva al dashboard y cierre la sesión de acceso del dashboard.
    """
    from flask import request

    app = ns["app"]
    regla_acceso = next(
        (rule for rule in app.url_map.iter_rules() if rule.rule == "/acceso"),
        None,
    )
    if regla_acceso is None:
        raise RuntimeError("No se ha localizado la ruta /acceso.")

    endpoint_acceso = regla_acceso.endpoint
    acceso_original = app.view_functions[endpoint_acceso]

    def acceso_dashboard(*args, **kwargs):
        if request.method == "POST" and _es_peticion_salida(request):
            return (
                "<!doctype html><html><head><meta charset='utf-8'>"
                "<title>Dashboard</title></head><body>"
                "<script>window.top.location.href='/dashboard/logout';</script>"
                "<p>Volviendo al dashboard…</p></body></html>"
            )
        return acceso_original(*args, **kwargs)

    app.view_functions[endpoint_acceso] = acceso_dashboard



def _animar_progreso_runtime(detener: threading.Event) -> None:
    """
    Suaviza únicamente el indicador visual durante la fase opaca y pesada de
    `base.ejecutar_runtime()`.

    No inspecciona, modifica ni interrumpe el runtime científico. El porcentaje
    es orientativo y queda limitado al 74 % hasta que la ejecución real retorna.
    """
    while not detener.wait(3.0):
        with estado_runtime.lock:
            if estado_runtime.estado != "inicializando":
                return

            actual = int(estado_runtime.progreso or 0)

        if actual < 30 or actual >= 74:
            continue

        # Incrementos pequeños y decrecientes para evitar saltos artificiales.
        if actual < 45:
            incremento = 3
        elif actual < 60:
            incremento = 2
        else:
            incremento = 1

        nuevo = min(74, actual + incremento)
        estado_runtime.actualizar(
            progreso=nuevo,
            detalle=(
                "Cargando el runtime científico real en segundo plano. "
                "Progreso orientativo; el dashboard permanece disponible."
            ),
        )


def inicializar_runtime() -> None:
    """
    Inicializa EXACTAMENTE la base funcional estable, pero en un hilo.

    El dashboard ya está servido mientras esta función trabaja. No se filtran
    los bloques AST que causaron las regresiones anteriores.
    """
    try:
        estado_runtime.actualizar(
            estado="inicializando",
            etapa="Localizando Notebook 05",
            detalle="Comprobando la fuente de runtime cerrada.",
            progreso=5,
        )

        # Misma raíz adicional y mismo selector de la versión estable validada.
        base.NOMBRES_RAIZ.update({"aplicacion_flujo"})
        base.seleccionar_runtime = seleccionar_runtime_robusto

        ruta_notebook = base.localizar_notebook()
        print(f"Notebook : {ruta_notebook}")
        print(f"SHA-256  : {base.sha256_archivo(ruta_notebook)}")
        print()
        print("Preparando runtime de la aplicación en segundo plano...")
        print("No se modifica ni se guarda el Notebook 05.")

        estado_runtime.actualizar(
            etapa="Analizando dependencias",
            detalle="Reconstruyendo el runtime necesario desde el Notebook 05.",
            progreso=15,
        )

        celdas = base.cargar_celdas_codigo(ruta_notebook)
        registros, celda_app = base.preparar_registros(celdas)
        seleccion = seleccionar_runtime_robusto(registros)

        celdas_runtime = sorted({reg.celda for reg in seleccion})
        await_runtime = [
            reg for reg in seleccion
            if base.contiene_await_nivel_superior(reg.ast_node)
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
        estado_runtime.actualizar(
            etapa="Inicializando motor predictivo y explicabilidad",
            detalle=(
                "Cargando las dependencias reales del flujo. "
                "El dashboard permanece disponible."
            ),
            progreso=30,
        )

        # Esta es la misma ejecución que ya funcionó E2E. Puede mostrar
        # validaciones SHAP/RAG heredadas, pero ya no bloquea la interfaz.
        detener_progreso = threading.Event()
        hilo_progreso = threading.Thread(
            target=_animar_progreso_runtime,
            args=(detener_progreso,),
            name="TFM_Runtime_Progress",
            daemon=True,
        )
        hilo_progreso.start()

        try:
            ns = base.ejecutar_runtime(seleccion, ruta_notebook)
        finally:
            detener_progreso.set()

        estado_runtime.actualizar(
            etapa="Reconstruyendo contratos",
            detalle="Validando Pydantic y rutas canónicas.",
            progreso=78,
        )

        base.reconstruir_modelos_pydantic(ns)
        base.fijar_rutas_canonicas_runtime(ns)
        _validar_runtime_funcional(ns)
        _adaptar_salida_aplicacion(ns)

        historial = comprobar_historial_directo(ns)

        estado_runtime.actualizar(
            etapa="Activando aplicación",
            detalle="Flask, LangGraph, RAG y servicios listos.",
            progreso=95,
        )

        with estado_runtime.lock:
            estado_runtime.ns = ns
            estado_runtime.app_real = ns["app"]
            estado_runtime.historial_inicial = historial

        estado_runtime.actualizar(
            estado="listo",
            etapa="Motor listo",
            detalle="La aplicación real ya puede generar informes.",
            progreso=100,
            error=None,
        )

        print("=" * 72)
        print("MOTOR LISTO")
        print("=" * 72)
        print(f"Dashboard : http://{base.HOST}:{base.PUERTO}/dashboard")
        print(f"Aplicación: http://{base.HOST}:{base.PUERTO}/acceso")
        print(f"Informes  : {len(historial)} compatibles")
        print("Acceso    : CAM")
        print("=" * 72)
        print()

    except Exception as exc:
        mensaje = str(exc)
        estado_runtime.actualizar(
            estado="error",
            etapa="Error de inicialización",
            detalle=mensaje,
            progreso=100,
            error=mensaje,
        )
        print()
        print("[ERROR] El motor de la demostración no ha podido inicializarse.")
        print(mensaje)
        print()
        traceback.print_exc()


def _pagina_motor() -> str:
    estado = estado_runtime.serializar()
    etapa = str(estado.get("etapa", "Inicializando motor"))
    detalle = str(estado.get("detalle", ""))
    progreso = int(estado.get("progreso", 0))

    return f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="2">
<title>Inicializando motor</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;background:#f5f7fa;color:#162033;
display:grid;place-items:center;min-height:100vh;margin:0;padding:24px}}
.card{{width:min(520px,100%);background:#fff;border:1px solid #e2e7ee;border-radius:15px;
padding:26px;box-shadow:0 12px 34px rgba(30,45,70,.09)}}
h2{{margin:0 0 8px}}p{{color:#667085;font-size:13px;line-height:1.55}}
.bar{{height:9px;background:#edf1f5;border-radius:99px;overflow:hidden;margin:18px 0}}
.bar span{{display:block;height:100%;width:{progreso}%;background:#1f4e79}}
small{{color:#98a2b3}}
</style>
</head>
<body>
<div class="card">
  <h2>{etapa}</h2>
  <p>{detalle}</p>
  <div class="bar"><span></span></div>
  <small>{progreso}% · La página se actualizará automáticamente.</small>
</div>
</body></html>"""


def crear_dashboard_frontal():
    from flask import (
        Flask,
        Response,
        jsonify,
        redirect,
        request,
        send_file,
        send_from_directory,
        session,
    )

    app = Flask("dashboard_demo_front")
    app.secret_key = secrets.token_hex(32)
    app.config.update(
        SESSION_COOKIE_NAME="dashboard_demo_session",
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
    )

    @app.get("/")
    def inicio():
        return redirect("/dashboard")

    @app.get("/dashboard")
    def dashboard():
        respuesta = send_file(RUTA_DASHBOARD, max_age=0)
        respuesta.headers["Cache-Control"] = (
            "no-store, no-cache, must-revalidate, max-age=0"
        )
        respuesta.headers["Pragma"] = "no-cache"
        respuesta.headers["Expires"] = "0"
        return respuesta

    @app.get("/dashboard/assets/<path:nombre>")
    def dashboard_asset(nombre):
        # Figuras estáticas exportadas de los notebooks ejecutados.
        return send_from_directory(RUTA_ASSETS, nombre, max_age=0)

    @app.get("/dashboard/api/status")
    def dashboard_status():
        return jsonify(estado_runtime.serializar())

    @app.get("/dashboard/api/informes")
    def dashboard_api_informes():
        return jsonify({
            "estado": "OK",
            "informes": _listar_informes_directos(),
        })

    @app.get("/dashboard/api/informe/<id_informe>")
    def dashboard_api_informe(id_informe):
        data = _buscar_informe_por_id(id_informe)
        if data is None:
            return jsonify({
                "estado": "ERROR",
                "error": "Informe no encontrado.",
            }), 404
        return jsonify(data)

    @app.get("/dashboard/logout")
    def dashboard_logout():
        # Fuerza una nueva pantalla de acceso real la próxima vez.
        session.pop("real_session_reset_done", None)

        respuesta = redirect("/dashboard")

        # La aplicación Flask recuperada utiliza su propia cookie de sesión.
        # Se elimina para no reutilizar una autenticación anterior del navegador.
        nombre_cookie_real = "session"
        if estado_runtime.app_real is not None:
            nombre_cookie_real = estado_runtime.app_real.config.get(
                "SESSION_COOKIE_NAME",
                "session",
            )

        respuesta.delete_cookie(
            nombre_cookie_real,
            path="/",
        )
        return respuesta

    @app.post("/dashboard/cerrar")
    def dashboard_cerrar():
        global servidor_dashboard

        servidor = servidor_dashboard
        if servidor is not None:
            threading.Timer(0.25, servidor.shutdown).start()

        return jsonify({"estado": "cerrando"})

    @app.route("/acceso", methods=["GET", "POST"])
    def acceso_controlado():
        """
        /acceso reproduce EXACTAMENTE la ruta real del Notebook 05.

        No se genera HTML alternativo para la autenticación.
        La única intervención del dashboard es borrar una sesión real anterior
        una vez por arranque/sesión de dashboard, de modo que el navegador no
        pueda saltarse la pantalla real por conservar cookies antiguas.
        """
        estado = estado_runtime.serializar()

        if estado["estado"] == "error":
            return (
                "<!doctype html><html><body style='font-family:Segoe UI,Arial;"
                "padding:32px'><h2>Motor no disponible</h2><p>"
                + str(estado.get("error") or "Error de inicialización.")
                + "</p><p><a href='/dashboard'>Volver al dashboard</a></p>"
                "</body></html>"
            ), 503

        if estado["estado"] != "listo":
            return _pagina_motor(), 503

        app_real = estado_runtime.app_real
        if app_real is None:
            return _pagina_motor(), 503

        # En el primer GET después de cada arranque, borrar cualquier cookie
        # de sesión de la aplicación REAL que el navegador conserve.
        if (
            request.method == "GET"
            and not session.get("real_session_reset_done", False)
        ):
            session["real_session_reset_done"] = True

            respuesta = redirect("/acceso")
            nombre_cookie_real = app_real.config.get(
                "SESSION_COOKIE_NAME",
                "session",
            )
            respuesta.delete_cookie(
                nombre_cookie_real,
                path="/",
            )
            return respuesta

        # A partir de aquí no se reinterpreta el formulario ni su HTML:
        # se sirve la interfaz real del Notebook 05 sin cambios visuales.
        return Response.from_app(app_real.wsgi_app, request.environ)

    @app.route(
        "/<path:ruta>",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    )
    def delegar_runtime(ruta):
        # Las rutas /dashboard/* anteriores tienen prioridad por especificidad.
        estado = estado_runtime.serializar()

        if estado["estado"] != "listo" or estado_runtime.app_real is None:
            if ruta.startswith("api/"):
                return jsonify({
                    "estado": "INICIALIZANDO"
                    if estado["estado"] != "error"
                    else "ERROR",
                    "motor": estado,
                }), 503

            return _pagina_motor(), 503

        return Response.from_app(
            estado_runtime.app_real.wsgi_app,
            request.environ,
        )

    return app


def main() -> int:
    global servidor_dashboard

    cabecera_dashboard_demo()

    import sys

    if Path(sys.prefix).name != "TFM_Agentes":
        print("[ERROR] El lanzador debe ejecutarse en el entorno TFM_Agentes.")
        print(f"        Entorno detectado: {Path(sys.prefix).name}")
        return 2

    if not base.RUTA_PROYECTO.exists():
        print(f"[ERROR] No existe el proyecto: {base.RUTA_PROYECTO}")
        return 2

    if not RUTA_DASHBOARD.exists():
        print(f"[ERROR] No existe: {RUTA_DASHBOARD}")
        return 2

    contenido = RUTA_DASHBOARD.read_text(encoding="utf-8")
    if MARCADOR_DASHBOARD not in contenido:
        print("[ERROR] dashboard_demo.html no corresponde a esta versión.")
        return 2

    from werkzeug.serving import make_server

    app_frontal = crear_dashboard_frontal()
    servidor_dashboard = make_server(
        base.HOST,
        base.PUERTO,
        app_frontal,
        threaded=True,
    )

    hilo_servidor = threading.Thread(
        target=servidor_dashboard.serve_forever,
        name="TFM_Dashboard_Front",
        daemon=False,
    )
    hilo_servidor.start()

    print("=" * 72)
    print("DASHBOARD DISPONIBLE")
    print("=" * 72)
    print(f"Dashboard : http://{base.HOST}:{base.PUERTO}/dashboard")
    print("Estado    : inicializando motor en segundo plano")
    print("Acceso    : interfaz real del Notebook 05 · credencial CAM")
    print("=" * 72)
    print()

    # Abrir el navegador ANTES de la inicialización pesada.
    base.abrir_navegador(
        f"http://{base.HOST}:{base.PUERTO}/dashboard"
    )

    hilo_runtime = threading.Thread(
        target=inicializar_runtime,
        name="TFM_Runtime_Background",
        daemon=True,
    )
    hilo_runtime.start()

    try:
        while hilo_servidor.is_alive():
            hilo_servidor.join(timeout=0.5)
    except KeyboardInterrupt:
        print("\nCerrando dashboard local...")
        servidor_dashboard.shutdown()
        hilo_servidor.join(timeout=10)

    print("Servidor detenido. Dashboard finalizado.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
