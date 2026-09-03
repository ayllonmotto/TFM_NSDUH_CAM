"""
Prueba integral del entorno TFM_Agentes.

Valida la configuración técnica necesaria para el Notebook 05 sin realizar
llamadas de pago a APIs externas ni modificar el entorno TFM_ML.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Literal, TypedDict

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from langchain_core.tools import tool
from langchain_mistralai import ChatMistralAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, START, StateGraph
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pydantic import BaseModel, Field, ValidationError, model_validator


# ============================================================
# CONFIGURACIÓN
# ============================================================

NOMBRE_ENTORNO = "TFM_Agentes"
PYTHON_ESPERADO = (3, 11)
KERNEL_ID = "tfm_agentes"
KERNEL_DISPLAY = "Python (TFM_Agentes)"
MODELO_EMBEDDINGS = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

RUTA_PROYECTO = Path.home() / "BD" / "TFM_NSDUH"
RUTA_ENV = RUTA_PROYECTO / ".env"
RUTA_KERNEL = (
    Path.home()
    / ".local"
    / "share"
    / "jupyter"
    / "kernels"
    / KERNEL_ID
    / "kernel.json"
)

VERSIONES_ESPERADAS = {
    "numpy": "2.4.6",
    "pandas": "3.0.5",
    "requests": "2.34.2",
    "pydantic": "2.13.4",
    "python-dotenv": "1.2.3",
    "flask": "3.1.3",
    "plotly": "6.9.0",
    "pypdf": "6.14.2",
    "ipykernel": "7.3.0",
    "torch": "2.13.0",
    "sentence-transformers": "5.7.0",
    "transformers": "5.15.1",
    "huggingface-hub": "1.28.0",
    "langchain": "1.3.16",
    "langchain-core": "1.6.0",
    "langgraph": "1.2.11",
    "langchain-mistralai": "1.1.6",
    "mcp": "1.29.1",
    "langchain-mcp-adapters": "0.3.2",
    "llama-index-core": "0.14.23",
    "llama-index-embeddings-huggingface": "0.7.0",
}

resultados: dict[str, bool] = {}
advertencias: list[str] = []


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def registrar(nombre: str, resultado: bool) -> None:
    resultados[nombre] = bool(resultado)
    estado = "OK" if resultado else "ERROR"
    print(f"{nombre:<46} {estado}")


def cabecera(texto: str) -> None:
    print()
    print("=" * 68)
    print(texto)
    print("=" * 68)


# ============================================================
# 1. ENTORNO, PYTHON Y VERSIONES
# ============================================================

cabecera("1. ENTORNO, PYTHON Y VERSIONES")

entorno_correcto = f"/envs/{NOMBRE_ENTORNO}/" in sys.executable
registrar("Intérprete TFM_Agentes", entorno_correcto)

python_correcto = sys.version_info[:2] == PYTHON_ESPERADO
registrar("Python 3.11", python_correcto)

versiones_ok = True

for paquete, esperada in VERSIONES_ESPERADAS.items():
    try:
        actual = version(paquete)
        coincide = actual == esperada
    except PackageNotFoundError:
        actual = "NO INSTALADO"
        coincide = False

    print(f"  {paquete:<38} {actual}")
    versiones_ok &= coincide

registrar("Versiones principales", versiones_ok)


# ============================================================
# 2. KERNEL JUPYTER
# ============================================================

cabecera("2. KERNEL JUPYTER")

kernel_ok = False

if RUTA_KERNEL.exists():
    datos_kernel = json.loads(RUTA_KERNEL.read_text(encoding="utf-8"))
    argv = datos_kernel.get("argv", [])
    display_name = datos_kernel.get("display_name", "")

    kernel_ok = (
        bool(argv)
        and f"/envs/{NOMBRE_ENTORNO}/bin/python" in argv[0]
        and display_name == KERNEL_DISPLAY
    )

registrar("Kernel Python (TFM_Agentes)", kernel_ok)


# ============================================================
# 3. CREDENCIALES Y SEGURIDAD
# ============================================================

cabecera("3. CREDENCIALES Y SEGURIDAD")

load_dotenv(RUTA_ENV)

env_existe = RUTA_ENV.exists()
registrar("Archivo .env", env_existe)

permisos_ok = (
    env_existe
    and (RUTA_ENV.stat().st_mode & 0o777) == 0o600
)
registrar("Permisos .env = 600", permisos_ok)

mistral_key_ok = bool(os.getenv("MISTRAL_API_KEY"))
hf_token_ok = bool(os.getenv("HF_TOKEN"))

registrar("MISTRAL_API_KEY definida", mistral_key_ok)
registrar("HF_TOKEN definido", hf_token_ok)

git_ignore = subprocess.run(
    ["git", "check-ignore", "-q", ".env"],
    cwd=RUTA_PROYECTO,
    check=False,
).returncode == 0
registrar(".env ignorado por Git", git_ignore)

git_versionado = subprocess.run(
    ["git", "ls-files", "--error-unmatch", ".env"],
    cwd=RUTA_PROYECTO,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    check=False,
).returncode == 0
registrar(".env no versionado", not git_versionado)


# ============================================================
# 4. PYTORCH, CUDA Y EMBEDDINGS
# ============================================================

cabecera("4. PYTORCH, CUDA Y EMBEDDINGS")

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

torch_ok = torch.__version__.startswith("2.13.0")
registrar("PyTorch", torch_ok)

cuda_disponible = torch.cuda.is_available()

if cuda_disponible:
    dispositivo = "cuda"
    tensor = torch.rand((256, 256), device="cuda")
    salida = tensor @ tensor
    torch.cuda.synchronize()
    cuda_ok = salida.device.type == "cuda" and bool(
        torch.isfinite(salida).all().item()
    )
    print("  GPU:", torch.cuda.get_device_name(0))
    print("  CUDA compilada:", torch.version.cuda)
else:
    dispositivo = "cpu"
    cuda_ok = True
    advertencias.append(
        "CUDA no disponible: se utilizará fallback automático a CPU."
    )

registrar("PyTorch CPU/GPU operativo", cuda_ok)

modelo_embeddings = SentenceTransformer(
    MODELO_EMBEDDINGS,
    device=dispositivo,
)

embeddings = modelo_embeddings.encode(
    [
        "Los valores SHAP no implican causalidad.",
        "El sistema tiene finalidad preventiva y no diagnóstica.",
    ],
    show_progress_bar=False,
    convert_to_numpy=True,
)

embeddings_ok = embeddings.shape == (2, 384)
registrar("sentence-transformers / embeddings", embeddings_ok)


# ============================================================
# 5. RAG CON LLAMAINDEX
# ============================================================

cabecera("5. LLAMAINDEX Y RAG")

Settings.embed_model = HuggingFaceEmbedding(
    model_name=MODELO_EMBEDDINGS,
    device=dispositivo,
)

documentos = [
    Document(
        text=(
            "Los valores SHAP expresan contribuciones al resultado del "
            "modelo y no relaciones causales."
        ),
        metadata={"fuente": "limitaciones_tfm", "tema": "shap"},
    ),
    Document(
        text=(
            "El sistema tiene finalidad preventiva y no constituye "
            "un diagnóstico clínico."
        ),
        metadata={"fuente": "uso_responsable", "tema": "limitaciones"},
    ),
]

indice = VectorStoreIndex.from_documents(documentos)
retriever = indice.as_retriever(similarity_top_k=1)
recuperados = retriever.retrieve(
    "¿Los valores SHAP permiten afirmar causalidad?"
)

rag_ok = (
    len(recuperados) == 1
    and recuperados[0].metadata.get("fuente") == "limitaciones_tfm"
)

registrar("LlamaIndex / recuperación semántica", rag_ok)


# ============================================================
# 6. PYDANTIC Y GUARDRAILS
# ============================================================

cabecera("6. PYDANTIC Y GUARDRAILS")


class ResultadoPredictivo(BaseModel):
    variable_objetivo: str = Field(min_length=1)
    probabilidad: float = Field(ge=0.0, le=1.0)
    umbral: float = Field(ge=0.0, le=1.0)
    clasificacion: Literal[0, 1]

    @model_validator(mode="after")
    def validar_clasificacion(self):
        esperada = int(self.probabilidad >= self.umbral)

        if self.clasificacion != esperada:
            raise ValueError(
                "Clasificación incoherente con probabilidad y umbral."
            )

        return self


valido = ResultadoPredictivo(
    variable_objetivo="OBJETIVO_PRUEBA",
    probabilidad=0.72,
    umbral=0.50,
    clasificacion=1,
)

pydantic_ok = valido.clasificacion == 1
registrar("Contrato Pydantic válido", pydantic_ok)

guardrail_ok = False

try:
    ResultadoPredictivo(
        variable_objetivo="OBJETIVO_PRUEBA",
        probabilidad=0.72,
        umbral=0.50,
        clasificacion=0,
    )
except ValidationError:
    guardrail_ok = True

registrar("Guardrail de coherencia", guardrail_ok)


# ============================================================
# 7. ADAPTADOR MISTRAL
# ============================================================

cabecera("7. ADAPTADOR MISTRAL")

try:
    llm = ChatMistralAI(
        model="mistral-small-latest",
        temperature=0,
    )
    mistral_ok = llm is not None and mistral_key_ok
except Exception:
    mistral_ok = False

registrar("Adaptador Mistral sin llamada API", mistral_ok)
print("  Llamada API realizada: False")


# ============================================================
# 8. TOOL LOCAL Y LANGGRAPH
# ============================================================

cabecera("8. TOOL LOCAL Y LANGGRAPH")


@tool
def consultar_resultado_predictivo(variable_objetivo: str) -> dict:
    """Devuelve un resultado predictivo sintético para prueba técnica."""
    return {
        "variable_objetivo": variable_objetivo,
        "probabilidad": 0.72,
        "umbral": 0.50,
        "clasificacion": 1,
    }


class EstadoAgente(TypedDict, total=False):
    variable_objetivo: str
    resultado_predictivo: dict
    resultado_validado: dict
    estado: str


def nodo_consulta(state: EstadoAgente):
    resultado = consultar_resultado_predictivo.invoke(
        {"variable_objetivo": state["variable_objetivo"]}
    )
    return {
        "resultado_predictivo": resultado,
        "estado": "resultado_recuperado",
    }


def nodo_validacion(state: EstadoAgente):
    validado = ResultadoPredictivo.model_validate(
        state["resultado_predictivo"]
    )
    return {
        "resultado_validado": validado.model_dump(),
        "estado": "validado",
    }


builder = StateGraph(EstadoAgente)
builder.add_node("consultar_prediccion", nodo_consulta)
builder.add_node("validar_resultado", nodo_validacion)
builder.add_edge(START, "consultar_prediccion")
builder.add_edge("consultar_prediccion", "validar_resultado")
builder.add_edge("validar_resultado", END)

grafo = builder.compile()
estado_final = grafo.invoke(
    {
        "variable_objetivo": "OBJETIVO_PRUEBA",
        "estado": "inicio",
    }
)

langgraph_ok = (
    estado_final.get("estado") == "validado"
    and estado_final.get("resultado_validado", {}).get("clasificacion") == 1
)

registrar("Tool local", "resultado_predictivo" in estado_final)
registrar("LangGraph con dos nodos", langgraph_ok)


# ============================================================
# 9. MCP CLIENT/SERVER Y LANGGRAPH
# ============================================================

cabecera("9. MCP CLIENT/SERVER Y LANGGRAPH")


class EstadoMCP(TypedDict):
    a: int
    b: int
    resultado_mcp: str


async def validar_mcp():
    codigo_servidor = """
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("TFM_MCP_Smoke_Test")


@mcp.tool()
def sumar(a: int, b: int) -> int:
    \"""Suma dos números enteros.\"""
    return a + b


if __name__ == "__main__":
    mcp.run(transport="stdio")
"""

    with tempfile.TemporaryDirectory() as directorio:
        ruta_servidor = Path(directorio) / "tfm_mcp_server.py"
        ruta_servidor.write_text(
            codigo_servidor,
            encoding="utf-8",
        )

        cliente_mcp = MultiServerMCPClient(
            {
                "tfm_test": {
                    "command": sys.executable,
                    "args": [str(ruta_servidor)],
                    "transport": "stdio",
                }
            }
        )

        herramientas = await cliente_mcp.get_tools()
        nombres = [herramienta.name for herramienta in herramientas]

        herramienta_sumar = next(
            (
                herramienta
                for herramienta in herramientas
                if herramienta.name == "sumar"
            ),
            None,
        )

        servidor_ok = herramienta_sumar is not None and "sumar" in nombres

        if not servidor_ok:
            return False, False, False

        resultado_tool = await herramienta_sumar.ainvoke(
            {
                "a": 12,
                "b": 30,
            }
        )

        tool_ok = "42" in str(resultado_tool)

        async def nodo_mcp(state: EstadoMCP):
            resultado = await herramienta_sumar.ainvoke(
                {
                    "a": state["a"],
                    "b": state["b"],
                }
            )
            return {
                "resultado_mcp": str(resultado),
            }

        builder_mcp = StateGraph(EstadoMCP)
        builder_mcp.add_node("ejecutar_tool_mcp", nodo_mcp)
        builder_mcp.add_edge(START, "ejecutar_tool_mcp")
        builder_mcp.add_edge("ejecutar_tool_mcp", END)

        grafo_mcp = builder_mcp.compile()

        resultado_grafo = await grafo_mcp.ainvoke(
            {
                "a": 15,
                "b": 27,
                "resultado_mcp": "",
            }
        )

        langgraph_mcp_ok = (
            "42" in resultado_grafo.get("resultado_mcp", "")
        )

        return servidor_ok, tool_ok, langgraph_mcp_ok


try:
    (
        mcp_stdio_ok,
        mcp_tool_ok,
        mcp_langgraph_ok,
    ) = asyncio.run(validar_mcp())
except Exception as error:
    print(f"  Error MCP: {type(error).__name__}: {error}")
    mcp_stdio_ok = False
    mcp_tool_ok = False
    mcp_langgraph_ok = False

registrar("MCP Client/Server stdio", mcp_stdio_ok)
registrar("MCP tool adaptada a LangChain", mcp_tool_ok)
registrar("MCP integrado en LangGraph", mcp_langgraph_ok)


# ============================================================
# 10. FLASK, HTTP Y JSON
# ============================================================

cabecera("10. FLASK, HTTP Y JSON")


class EntradaHTTP(BaseModel):
    identificador: str = Field(min_length=1)
    valor: float = Field(ge=0.0, le=1.0)


app = Flask(__name__)


@app.get("/api/v1/health")
def health():
    return jsonify({"estado": "OK", "servicio": "TFM_Agentes"}), 200


@app.post("/api/v1/prueba")
def prueba():
    try:
        entrada = EntradaHTTP.model_validate(request.get_json())
        return jsonify(
            {
                "estado": "OK",
                "identificador": entrada.identificador,
                "valor_recibido": entrada.valor,
            }
        ), 200
    except ValidationError:
        return jsonify({"estado": "ERROR_VALIDACION"}), 422


cliente = app.test_client()
r_health = cliente.get("/api/v1/health")
r_valida = cliente.post(
    "/api/v1/prueba",
    json={"identificador": "TFM_PRUEBA", "valor": 0.72},
)
r_invalida = cliente.post(
    "/api/v1/prueba",
    json={"identificador": "TFM_PRUEBA", "valor": 1.50},
)

flask_ok = (
    r_health.status_code == 200
    and r_valida.status_code == 200
    and r_invalida.status_code == 422
    and r_health.is_json
    and r_valida.is_json
)

registrar("Flask / HTTP / JSON", flask_ok)


# ============================================================
# 11. AISLAMIENTO TFM_AGENTES / TFM_ML
# ============================================================

cabecera("11. AISLAMIENTO TFM_AGENTES / TFM_ML")

modulos_predictivos = ["tensorflow", "xgboost", "shap"]

agentes_sin_predictivos = all(
    importlib.util.find_spec(modulo) is None
    for modulo in modulos_predictivos
)
registrar(
    "TFM_Agentes sin TensorFlow/XGBoost/SHAP",
    agentes_sin_predictivos,
)

codigo_tfm_ml = """
import importlib.util
import json
import sys

modulos = ["tensorflow", "xgboost", "shap"]

print(json.dumps({
    "python": sys.executable,
    "modulos": {
        modulo: importlib.util.find_spec(modulo) is not None
        for modulo in modulos
    },
    "estado": "OK"
}))
"""

try:
    proceso = subprocess.run(
        [
            str(Path.home() / "miniconda3" / "bin" / "conda"),
            "run",
            "-n",
            "TFM_ML",
            "python",
            "-c",
            codigo_tfm_ml,
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )

    lineas = [
        linea.strip()
        for linea in proceso.stdout.splitlines()
        if linea.strip()
    ]

    resultado_ml = json.loads(lineas[-1])

    bridge_ok = (
        "/envs/TFM_ML/" in resultado_ml["python"]
        and all(resultado_ml["modulos"].values())
        and resultado_ml["estado"] == "OK"
    )
except Exception:
    bridge_ok = False

registrar("Bridge JSON hacia TFM_ML", bridge_ok)


# ============================================================
# 12. CONSISTENCIA DE DEPENDENCIAS
# ============================================================

cabecera("12. CONSISTENCIA DE DEPENDENCIAS")

pip_check = subprocess.run(
    [sys.executable, "-m", "pip", "check"],
    capture_output=True,
    text=True,
    check=False,
)

pip_ok = pip_check.returncode == 0
registrar("pip check", pip_ok)

if pip_check.stdout.strip():
    print(" ", pip_check.stdout.strip())


# ============================================================
# RESULTADO FINAL
# ============================================================

cabecera("RESULTADO FINAL")

for nombre, resultado in resultados.items():
    print(f"{nombre:<46} {resultado}")

if advertencias:
    print()
    print("ADVERTENCIAS:")
    for advertencia in advertencias:
        print("-", advertencia)

print()
resultado_global = all(resultados.values())
print("SMOKE TEST SUPERADO:", resultado_global)

if not resultado_global:
    sys.exit(1)
