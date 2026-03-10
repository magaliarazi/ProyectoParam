#!/bin/bash

INPUT_LIST="molecules.txt"
OUTPUT_DIR="sdf"

mkdir -p "$OUTPUT_DIR"

while read mol; do
    # Saltear líneas vacías
    [ -z "$mol" ] && continue

    echo "Descargando $mol ..."

    wget -q \
      "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/${mol}/record/SDF/?record_type=3d" \
      -O "${OUTPUT_DIR}/${mol}.sdf"

    # Verificación básica
    if [ ! -s "${OUTPUT_DIR}/${mol}.sdf" ]; then
        echo "  ⚠️  Falló la descarga de $mol"
        rm -f "${OUTPUT_DIR}/${mol}.sdf"
    fi
done < "$INPUT_LIST"

echo "Descarga completada."
