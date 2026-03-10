#!/bin/bash

mkdir -p acpype_out

for f in mol2/*.mol2; do
    name=$(basename "$f" .mol2)
    echo "Procesando $name"

    acpype -i "$f" -b "$name"

    if [ -d "${name}.acpype" ]; then
        mv "${name}.acpype" acpype_out/
    else
        echo "⚠️ Falló $name"
    fi
done
