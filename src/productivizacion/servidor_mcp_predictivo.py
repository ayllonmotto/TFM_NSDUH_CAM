# =============================================================================
# SERVIDOR MCP PREDICTIVO
# =============================================================================
import json
import os
import subprocess

from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP


# =============================================================================
# CONFIGURACIÓN
# =============================================================================
ruta_proyecto = Path.home() / 'BD' / 'TFM_NSDUH'
ruta_servicio_predictivo = (
    ruta_proyecto / 'src' / 'productivizacion' / 'servicio_predictivo.py'
)
ruta_conda = Path(
    os.environ.get('CONDA_EXE', Path.home() / 'miniconda3' / 'bin' / 'conda')
)

mcp = FastMCP('TFM NSDUH Predictivo')


# =============================================================================
# BRIDGE
# =============================================================================
def ejecutar_bridge(solicitud):
    proceso = subprocess.run(
        [
            str(ruta_conda), 'run', '--no-capture-output', '-n', 'TFM_ML',
            'python', str(ruta_servicio_predictivo)
        ],
        input=json.dumps(solicitud, ensure_ascii=False, allow_nan=False),
        capture_output=True, text=True, check=False, timeout=180
    )

    if proceso.returncode != 0:
        raise RuntimeError(proceso.stderr)

    lineas = [linea.strip() for linea in proceso.stdout.splitlines() if linea.strip()]

    if not lineas:
        raise RuntimeError('El servicio predictivo no ha devuelto ninguna respuesta.')

    return json.loads(lineas[-1])


# =============================================================================
# TOOL MCP
# =============================================================================
@mcp.tool()
def predecir_indicadores(registro: dict[str, Any]) -> str:
    """Obtiene las cuatro predicciones definitivas a partir de un registro NSDUH."""
    respuesta = ejecutar_bridge({'operacion': 'predict', 'registro': registro})

    if respuesta.get('estado') != 'OK':
        raise RuntimeError(respuesta.get('mensaje', 'Error en el servicio predictivo.'))

    return json.dumps(respuesta, ensure_ascii=False, allow_nan=False)


if __name__ == '__main__':
    mcp.run(transport='stdio')