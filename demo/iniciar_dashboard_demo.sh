#!/usr/bin/env bash
set -euo pipefail

RUTA_DEMO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$RUTA_DEMO"

CONDA_BIN="${CONDA_EXE:-$HOME/miniconda3/bin/conda}"

if [[ ! -x "$CONDA_BIN" ]]; then
  echo "[ERROR] No se ha localizado Conda en: $CONDA_BIN"
  exit 1
fi

echo "============================================================"
echo "TFM NSDUH 2024 - DASHBOARD DEMO"
echo "============================================================"
echo "Ruta: $RUTA_DEMO"
echo "URL : http://127.0.0.1:5000/dashboard"
echo
echo "El dashboard se abrirá inmediatamente."
echo "El motor real se inicializará en segundo plano."
echo "Para finalizar, utiliza 'Cerrar demo' o pulsa Ctrl+C."
echo "============================================================"
echo

exec "$CONDA_BIN" run --no-capture-output -n TFM_Agentes \
  python "$RUTA_DEMO/dashboard_demo.py"
