#!/bin/bash

mkdir -p mol2

for f in sdf/*.sdf; do
  name=$(basename "$f" .sdf)
  obabel "$f" -O "mol2/${name}.mol2" -h
done
