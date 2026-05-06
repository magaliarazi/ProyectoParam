"""
extract_itp_targets.py
----------------------
Lee pares de archivos .mol2 y .itp (mismo nombre base) de una carpeta
y genera un CSV con atom_type y charge por átomo, emparejados por nombre de átomo.

Uso:
    python extract_itp_targets.py --input carpeta/ --output targets.csv

Dependencias:
    pip install pandas
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def parse_itp_atoms(itp_path: Path) -> dict:
    """
    Parsea el bloque [ atoms ] del .itp.
    Devuelve dict: {atom_name: {"atomtype": str, "charge": float}}
    """
    atoms = {}
    in_atoms_block = False

    with open(itp_path, "r") as f:
        for line in f:
            line = line.strip()

            # Detectar entrada al bloque [ atoms ]
            if line.startswith("[ atoms ]") or line == "[ atoms ]":
                in_atoms_block = True
                continue

            # Salir del bloque al encontrar otra sección
            if in_atoms_block and line.startswith("["):
                break

            # Ignorar comentarios y líneas vacías
            if not line or line.startswith(";"):
                continue

            if in_atoms_block:
                parts = line.split()
                # formato: nr  type  resnr  resid  atom  cgnr  charge  mass
                if len(parts) >= 7:
                    atom_name = parts[4]
                    atomtype  = parts[1]
                    charge    = float(parts[6])
                    atoms[atom_name] = {"atomtype": atomtype, "charge": charge}

    return atoms


def parse_mol2_atom_names(mol2_path: Path) -> list:
    """
    Parsea el bloque @<TRIPOS>ATOM del .mol2.
    Devuelve lista de dicts: [{atom_id, atom_name}]
    """
    atoms = []
    in_atom_block = False

    with open(mol2_path, "r") as f:
        for line in f:
            line = line.rstrip()
            if line.startswith("@<TRIPOS>ATOM"):
                in_atom_block = True
                continue
            if line.startswith("@<TRIPOS>") and in_atom_block:
                break
            if in_atom_block and line.strip():
                parts = line.split()
                if len(parts) >= 2:
                    atom_id   = int(parts[0])
                    atom_name = parts[1]
                    atoms.append({"atom_id": atom_id, "atom_name": atom_name})

    return atoms


def process_molecule(mol2_path: Path, itp_path: Path) -> list:
    """
    Empareja átomos del .mol2 con tipos y cargas del .itp por nombre de átomo.
    Devuelve lista de dicts listos para el CSV.
    """
    mol_name = mol2_path.stem

    itp_atoms  = parse_itp_atoms(itp_path)
    mol2_atoms = parse_mol2_atom_names(mol2_path)

    rows = []
    unmatched = []

    for atom in mol2_atoms:
        atom_name = atom["atom_name"]
        itp_data  = itp_atoms.get(atom_name)

        if itp_data is None:
            unmatched.append(atom_name)
            atomtype = None
            charge   = None
        else:
            atomtype = itp_data["atomtype"]
            charge   = itp_data["charge"]

        rows.append({
            "molecule":  mol_name,
            "atom_id":   atom["atom_id"],
            "atom_name": atom_name,
            "atomtype":  atomtype,
            "charge":    charge,
        })

    if unmatched:
        print(f"  [WARN] {mol2_path.name}: {len(unmatched)} átomo(s) sin match en .itp → {unmatched}")

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Extrae atomtype y charge del .itp y los empareja con átomos del .mol2"
    )
    parser.add_argument("--input",  "-i", required=True, help="Carpeta con archivos .mol2 e .itp")
    parser.add_argument("--output", "-o", default="targets.csv", help="Archivo CSV de salida")
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.is_dir():
        sys.exit(f"Error: '{input_dir}' no es una carpeta válida.")

    mol2_files = sorted(input_dir.glob("*.mol2"))
    if not mol2_files:
        sys.exit(f"No se encontraron archivos .mol2 en '{input_dir}'.")

    print(f"Encontrados {len(mol2_files)} archivos .mol2 en '{input_dir}'")

    all_rows = []
    skipped  = []

    for mol2_path in mol2_files:
        itp_path = mol2_path.with_suffix(".itp")

        if not itp_path.exists():
            print(f"  [SKIP] Sin .itp para: {mol2_path.name}")
            skipped.append(mol2_path.name)
            continue

        print(f"  Procesando: {mol2_path.name} ...", end=" ")
        rows = process_molecule(mol2_path, itp_path)
        all_rows.extend(rows)
        print(f"{len(rows)} átomos")

    if not all_rows:
        sys.exit("No se pudo extraer ningún átomo. Verificá los archivos.")

    df = pd.DataFrame(all_rows, columns=[
        "molecule", "atom_id", "atom_name", "atomtype", "charge"
    ])

    output_path = Path(args.output)
    df.to_csv(output_path, index=False)

    print(f"\n✓ CSV guardado en: {output_path}")
    print(f"  {len(df)} filas · {df['molecule'].nunique()} moléculas")
    if skipped:
        print(f"  Moléculas sin .itp ({len(skipped)}): {skipped}")


if __name__ == "__main__":
    main()