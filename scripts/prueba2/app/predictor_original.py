"""
predictor.py
------------
Lógica de predicción: toma un .mol2, extrae features, aplica
preprocesamiento y predice atomtype y charge con los modelos AA y UA.

Importa las funciones de extracción de features directamente desde
extract_dataset_v3.py para garantizar consistencia con el pipeline
de entrenamiento.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load
from sklearn.feature_extraction import FeatureHasher

# =============================================================================
# IMPORTAR FUNCIONES DE EXTRACCIÓN DESDE EL PIPELINE
# =============================================================================

PIPELINE_DIR = Path("/home/marazi/proyectoParam/scripts/prueba2/version2(calude)")
sys.path.append(str(PIPELINE_DIR))

from extract_dataset_v3 import (
    parse_mol2,
    build_rdkit_mol,
    bond_length,
    angle_deg,
    is_planar_geometry,
    ATOMIC_MASSES,
    PLANAR_TYPES,
    ELECTRONEGATIVE,
    ELEMENT_NORMALIZE,
    BOND_TYPE_MAP,
)

# =============================================================================
# CONFIG — rutas a modelos y artifacts
# =============================================================================

BASE_DIR   = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"

CLF_AA    = MODELS_DIR / "clf_AA.joblib"
REG_AA    = MODELS_DIR / "reg_AA.joblib"
CLF_UA    = MODELS_DIR / "clf_UA.joblib"
REG_UA    = MODELS_DIR / "reg_UA.joblib"
ARTIFACTS = MODELS_DIR / "preprocessing_artifacts.json"

HASH_N_FEATURES = 32
COLS_TO_DROP    = ["degree_of_unsat", "n_dihedrals", "bonds_single", "tripos_type"]
BOOL_COLS       = ["is_planar", "is_in_ring", "is_aromatic"]
ID_COLS         = ["molecule", "atom_id", "atom_name", "bonded_to_element"]
TARGET_COLS     = ["atomtype", "charge"]


# =============================================================================
# EXTRACCIÓN DE FEATURES
# (usa las funciones importadas de extract_dataset_v3.py)
# =============================================================================

def extract_features(mol2_path):
    """
    Extrae features atómicas del .mol2 usando las mismas funciones
    del pipeline de entrenamiento (extract_dataset_v3.py).
    No requiere .itp ya que en predicción no hay targets.
    """
    atoms, bonds = parse_mol2(Path(mol2_path))

    graph = {aid: [] for aid in atoms}
    for a1, a2, btype in bonds:
        if a1 in graph and a2 in graph:
            graph[a1].append((a2, btype))
            graph[a2].append((a1, btype))

    # n_dihedrals
    dihedral_counts = {aid: 0 for aid in atoms}
    for a1, a2, _ in bonds:
        if a1 not in graph or a2 not in graph:
            continue
        for (a, _) in graph[a1]:
            if a == a2:
                continue
            for (d, _) in graph[a2]:
                if d == a1:
                    continue
                for idx in (a, a1, a2, d):
                    if idx in dihedral_counts:
                        dihedral_counts[idx] += 1

    # RDKit
    mol, id_to_idx = build_rdkit_mol(atoms, bonds)
    has_rdkit     = mol is not None
    has_gasteiger = False
    ring_info     = None

    if has_rdkit:
        try:
            from rdkit.Chem import rdPartialCharges
            rdPartialCharges.ComputeGasteigerCharges(mol)
            has_gasteiger = True
        except Exception:
            pass
        ring_info = mol.GetRingInfo()

    rows = []
    for atom_id in sorted(atoms):
        atom        = atoms[atom_id]
        tripos_type = atom["tripos_type"]
        atom_name   = atom["atom_name"]
        element     = atom["element"]
        xyz_i       = atom["xyz"]
        mass        = ATOMIC_MASSES.get(element, 0.0)
        neighbors   = graph[atom_id]
        coordination= len(neighbors)

        neighbor_info    = []
        bond_lengths     = []
        neighbor_xyzs    = []
        elem_counts      = {"C": 0, "H": 0, "O": 0, "N": 0}
        bond_type_counts = {"single": 0, "double": 0, "aromatic": 0}
        n_electroneg_1   = 0
        bonded_elements  = []

        for nb_id, btype_str in neighbors:
            nb      = atoms[nb_id]
            nb_elem = nb["element"]
            nb_xyz  = nb["xyz"]
            bonded_elements.append(nb_elem)
            bond_lengths.append(bond_length(xyz_i, nb_xyz))
            neighbor_xyzs.append(nb_xyz)
            if nb_elem in elem_counts:
                elem_counts[nb_elem] += 1
            if nb_elem in ELECTRONEGATIVE:
                n_electroneg_1 += 1
            btype_norm = btype_str.lower()
            if btype_norm == "ar":
                bond_type_counts["aromatic"] += 1
            elif btype_norm in ("2", "3"):
                bond_type_counts["double"] += 1
            else:
                bond_type_counts["single"] += 1
            neighbor_info.append(f"{nb_elem}{btype_str}")

        bonded_to_element = (bonded_elements[0] if coordination == 1 and bonded_elements
                             else ",".join(sorted(bonded_elements)) if coordination > 1 else "")
        neighbor_config = "-".join(sorted(neighbor_info))
        avg_bond_len    = float(np.mean(bond_lengths)) if bond_lengths else 0.0

        seen2 = set()
        neighbors2_elems = []
        for nb_id, _ in neighbors:
            for nb2_id, _ in graph[nb_id]:
                if nb2_id != atom_id and nb2_id not in seen2:
                    seen2.add(nb2_id)
                    neighbors2_elems.append(atoms[nb2_id]["element"])
        neighbor2_config = "-".join(sorted(neighbors2_elems))
        n_electroneg_2   = sum(1 for e in neighbors2_elems if e in ELECTRONEGATIVE)

        nb_list = [nb_id for nb_id, _ in neighbors]
        angles  = [angle_deg(atoms[nb_list[j]]["xyz"], xyz_i, atoms[nb_list[k]]["xyz"])
                   for j in range(len(nb_list)) for k in range(j+1, len(nb_list))]
        avg_angle = float(np.mean(angles)) if angles else 0.0

        if tripos_type in PLANAR_TYPES:
            planar = True
        elif coordination >= 3:
            planar = is_planar_geometry(xyz_i, neighbor_xyzs)
        else:
            planar = False

        # RDKit features
        if has_rdkit and id_to_idx:
            rdkit_idx = id_to_idx.get(atom_id)
            if rdkit_idx is not None:
                rd_atom       = mol.GetAtomWithIdx(rdkit_idx)
                is_in_ring    = ring_info.NumAtomRings(rdkit_idx) > 0
                atom_rings    = [r for r in ring_info.AtomRings() if rdkit_idx in r]
                ring_size     = min(len(r) for r in atom_rings) if atom_rings else 0
                is_aromatic   = rd_atom.GetIsAromatic()
                formal_charge = rd_atom.GetFormalCharge()
                if has_gasteiger:
                    try:
                        g = float(rd_atom.GetDoubleProp("_GasteigerCharge"))
                        gasteiger_charge = 0.0 if (math.isnan(g) or math.isinf(g)) else g
                    except Exception:
                        gasteiger_charge = 0.0
                else:
                    gasteiger_charge = 0.0
            else:
                is_in_ring = is_aromatic = False
                ring_size = formal_charge = 0
                gasteiger_charge = 0.0
        else:
            is_in_ring    = tripos_type in {"C.ar", "N.ar"}
            ring_size     = 6 if is_in_ring else 0
            is_aromatic   = is_in_ring
            formal_charge = 0
            gasteiger_charge = 0.0

        degree_of_unsat = bond_type_counts["double"] + bond_type_counts["aromatic"]

        rows.append({
            "molecule":                atom["atom_name"],  # temporal, se sobreescribe
            "atom_id":                 atom_id,
            "atom_name":               atom_name,
            "tripos_type":             tripos_type,
            "element":                 element,
            "mass":                    round(mass, 4),
            "coordination":            coordination,
            "avg_bond_len_A":          round(avg_bond_len, 4),
            "avg_angle_deg":           round(avg_angle, 4),
            "is_planar":               planar,
            "C_count":                 elem_counts["C"],
            "H_count":                 elem_counts["H"],
            "O_count":                 elem_counts["O"],
            "N_count":                 elem_counts["N"],
            "neighbor_config":         neighbor_config,
            "neighbor2_config":        neighbor2_config,
            "n_dihedrals":             dihedral_counts[atom_id],
            "n_electroneg_neighbors":  n_electroneg_1,
            "n_electroneg_neighbors2": n_electroneg_2,
            "bonds_single":            bond_type_counts["single"],
            "bonds_double":            bond_type_counts["double"],
            "bonds_aromatic":          bond_type_counts["aromatic"],
            "is_in_ring":              is_in_ring,
            "ring_size":               ring_size,
            "is_aromatic":             is_aromatic,
            "formal_charge":           formal_charge,
            "degree_of_unsat":         degree_of_unsat,
            "gasteiger_charge":        round(gasteiger_charge, 6),
            "bonded_to_element":       bonded_to_element,
        })

    return pd.DataFrame(rows)


# =============================================================================
# PREPROCESAMIENTO
# =============================================================================

def preprocess_for_prediction(df, artifacts, scheme):
    """Aplica el mismo preprocesamiento que en entrenamiento."""
    df = df.copy()

    for col in COLS_TO_DROP:
        if col in df.columns:
            df = df.drop(columns=[col])

    for col in BOOL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(int)

    ohe_cats = artifacts[scheme]["ohe_categories"]
    for col, cats in ohe_cats.items():
        if col not in df.columns:
            continue
        for cat in cats:
            df[f"{col}_{cat}"] = (df[col] == cat).astype(int)
        df = df.drop(columns=[col])

    hash_cols = artifacts[scheme]["hash_cols"]
    for col, n_features in hash_cols.items():
        if col not in df.columns:
            continue
        hasher    = FeatureHasher(n_features=n_features, input_type="string")
        hashed    = hasher.transform(df[col].fillna("").apply(lambda x: [x])).toarray()
        col_names = [f"{col}_h{i}" for i in range(n_features)]
        hashed_df = pd.DataFrame(hashed, columns=col_names, index=df.index, dtype=int)
        df = pd.concat([df.drop(columns=[col]), hashed_df], axis=1)

    return df


# =============================================================================
# SPLIT AA / UA
# =============================================================================

def split_AA_UA(df):
    POLAR = {"N", "O"}

    def is_AA(row):
        elem   = row["element"]
        bonded = row["bonded_to_element"]
        if elem not in ("C", "H"):
            return True
        if elem == "C":
            return True
        if elem == "H" and bonded in ("C", *POLAR):
            return True
        return False

    def is_UA(row):
        elem   = row["element"]
        bonded = row["bonded_to_element"]
        if elem not in ("C", "H"):
            return True
        if elem == "C":
            return True
        if elem == "H" and bonded in POLAR:
            return True
        return False

    df_AA = df[df.apply(is_AA, axis=1)].copy()
    df_UA = df[df.apply(is_UA, axis=1)].copy()
    return df_AA, df_UA


# =============================================================================
# PREDICCIÓN PRINCIPAL
# =============================================================================

def predict_molecule(mol2_path, original_name=None):
    clf_aa = load(CLF_AA)
    reg_aa = load(REG_AA)
    clf_ua = load(CLF_UA)
    reg_ua = load(REG_UA)

    with open(ARTIFACTS) as f:
        artifacts = json.load(f)

    # Nombre de la molécula desde el archivo original
    mol_name = Path(original_name).stem if original_name else Path(mol2_path).stem

    # Extraer features usando funciones del pipeline
    df_raw = extract_features(mol2_path)
    df_raw["molecule"] = mol_name

    # Split AA / UA
    df_aa_raw, df_ua_raw = split_AA_UA(df_raw)

    results = {}

    for scheme, df_scheme, clf, reg in [
        ("AA", df_aa_raw, clf_aa, reg_aa),
        ("UA", df_ua_raw, clf_ua, reg_ua),
    ]:
        id_cols_scheme = df_scheme[["molecule", "atom_id", "atom_name", "element"]].copy()

        df_proc = preprocess_for_prediction(df_scheme, artifacts, scheme)

        expected_cols = list(clf.feature_names_in_) if hasattr(clf, "feature_names_in_") else []
        if expected_cols:
            for col in expected_cols:
                if col not in df_proc.columns:
                    df_proc[col] = 0
            df_proc = df_proc[expected_cols]

        atomtypes = clf.predict(df_proc)
        charges   = reg.predict(df_proc)

        result_df = id_cols_scheme.copy()
        result_df["atomtype_predicho"] = atomtypes
        result_df["charge_predicha"]   = [round(float(c), 4) for c in charges]

        results[scheme] = result_df.to_dict(orient="records")

    return {
        "molecule": mol_name,
        "AA":       results["AA"],
        "UA":       results["UA"],
    }