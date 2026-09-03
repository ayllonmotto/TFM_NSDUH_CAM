#!/usr/bin/env bash

set -euo pipefail

ENV_NAME="${1:-TFM_ML}"

echo "============================================================"
echo "CONFIGURACIÓN DE ENLACES CUDA — $ENV_NAME"
echo "============================================================"

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: no se encuentra el comando conda."
    exit 1
fi

ENV_PREFIX="$(
    conda run -n "$ENV_NAME" \
        python -c 'import sys; print(sys.prefix)' |
        tail -n 1
)"

SITE_PACKAGES="$(
    conda run -n "$ENV_NAME" python -c '
import site
paths = site.getsitepackages()
if not paths:
    raise RuntimeError("No se encontró site-packages.")
print(paths[0])
' | tail -n 1
)"

TF_DIR="$SITE_PACKAGES/tensorflow"
NVIDIA_DIR="$SITE_PACKAGES/nvidia"

echo "Entorno:       $ENV_PREFIX"
echo "site-packages: $SITE_PACKAGES"
echo "TensorFlow:    $TF_DIR"
echo "NVIDIA:        $NVIDIA_DIR"

test -d "$ENV_PREFIX"
test -d "$TF_DIR"
test -d "$NVIDIA_DIR"

PTXAS="$(
    find "$NVIDIA_DIR/cuda_nvcc/bin" \
        -type f \
        -name ptxas \
        -print \
        -quit
)"

if [ -z "$PTXAS" ]; then
    echo "ERROR: no se encontró ptxas."
    exit 1
fi

LIB_COUNT="$(
    find "$NVIDIA_DIR" \
        -path '*/lib/*.so*' \
        \( -type f -o -type l \) |
        wc -l
)"

if [ "$LIB_COUNT" -eq 0 ]; then
    echo "ERROR: no se encontraron bibliotecas NVIDIA."
    exit 1
fi

echo "Bibliotecas localizadas: $LIB_COUNT"
echo "ptxas: $PTXAS"

LINK_COUNT=0

while IFS= read -r -d '' LIBRARY; do
    DESTINATION="$TF_DIR/$(basename "$LIBRARY")"

    if [ -e "$DESTINATION" ] && [ ! -L "$DESTINATION" ]; then
        echo "ERROR: existe un archivo real en:"
        echo "$DESTINATION"
        exit 1
    fi

    ln -sfn "$LIBRARY" "$DESTINATION"
    LINK_COUNT=$((LINK_COUNT + 1))
done < <(
    find "$NVIDIA_DIR" \
        -path '*/lib/*.so*' \
        \( -type f -o -type l \) \
        -print0
)

ln -sfn "$PTXAS" "$ENV_PREFIX/bin/ptxas"

echo "Enlaces procesados: $LINK_COUNT"
echo "Enlace ptxas: $ENV_PREFIX/bin/ptxas"

CRITICAL_LIBS=(
    "libcudart.so.12"
    "libcublas.so.12"
    "libcublasLt.so.12"
    "libcudnn.so.9"
    "libcufft.so.11"
    "libcusolver.so.11"
    "libcusparse.so.12"
    "libnccl.so.2"
    "libnvJitLink.so.12"
)

echo
echo "Bibliotecas críticas:"

for LIBRARY in "${CRITICAL_LIBS[@]}"; do
    DESTINATION="$TF_DIR/$LIBRARY"

    if [ ! -e "$DESTINATION" ]; then
        echo "ERROR: no se resuelve $LIBRARY"
        exit 1
    fi

    printf "OK  %-22s -> %s\n" \
        "$LIBRARY" \
        "$(readlink -f "$DESTINATION")"
done

if [ ! -x "$ENV_PREFIX/bin/ptxas" ]; then
    echo "ERROR: ptxas no es ejecutable."
    exit 1
fi

echo
echo "============================================================"
echo "ENLACES CUDA CONFIGURADOS CORRECTAMENTE"
echo "============================================================"
