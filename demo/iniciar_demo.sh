#!/usr/bin/env bash
set -euo pipefail

PROYECTO="$HOME/BD/TFM_NSDUH"
PYTHON="$HOME/miniconda3/envs/TFM_Agentes/bin/python"
DEMO="$PROYECTO/demo/demo_tfm.py"

if [[ ! -x "$PYTHON" ]]; then
    echo "[ERROR] No se encuentra Python de TFM_Agentes:"
    echo "        $PYTHON"
    exit 1
fi

if [[ ! -f "$DEMO" ]]; then
    echo "[ERROR] No se encuentra:"
    echo "        $DEMO"
    exit 1
fi

cd "$PROYECTO"
exec "$PYTHON" "$DEMO"
