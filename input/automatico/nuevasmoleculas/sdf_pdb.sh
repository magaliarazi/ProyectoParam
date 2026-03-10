#!/bin/bash

INPUT_DIR=~/proyectoParam/input/automatico/nuevasmoleculas/sdf
OUTPUT_DIR=~/proyectoParam/input/automatico/nuevasmoleculas/pdb

mkdir -p "$OUTPUT_DIR"

# Iterar sobre todos los SDF
for sdf in "$INPUT_DIR"/*.sdf; do
    base=$(basename "$sdf" .sdf)
    echo "Procesando $base ..."

    # Crear carpeta por molécula
    mol_dir="$OUTPUT_DIR/$base"
    mkdir -p "$mol_dir"

    # 1️⃣ Convertir SDF -> PDB (generar coordenadas 3D)
    timeout 60 obabel "$sdf" -O "$mol_dir/$base.pdb" --gen3d
    if [ $? -ne 0 ]; then
        echo " ⚠️ Error al generar PDB para $base"
        continue
    fi

    # 2️⃣ Generar SMILES
    obabel "$sdf" -O "$mol_dir/$base.smi"
    if [ $? -ne 0 ]; then
        echo " ⚠️ Error al generar SMILES para $base"
        continue
    fi

    echo " ✅ $base listo"
done

echo "Todos los SDF se han convertido a PDB y SMILES."
