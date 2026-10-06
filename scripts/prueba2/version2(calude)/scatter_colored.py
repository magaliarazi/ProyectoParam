"""
scatter_colored.py
------------------
Genera scatters de charge real vs predicha coloreando los átomos
según si su atomtype es conocido (visto en entrenamiento) o nuevo.

Uso:
    python scatter_colored.py \
        --results_dir validation_external_E3/ \
        --output_dir validation_external_E3/figuras_coloreadas/
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Atomtypes conocidos (vocabulario de entrenamiento AA y UA)
KNOWN_ATOMTYPES = {
    "C", "HC", "H", "OA", "OE", "OM", "N", "O",
    "CH1", "CH2", "CH3", "CL", "F", "NT", "HS14", "S"
}

# Atomtypes nuevos (no vistos en entrenamiento)
NEW_ATOMTYPES = {"CPos", "OAlc", "OEOpt"}


def plot_scatter_colored(df, title, out_path, scheme):
    y_true = df["charge_real"].values.astype(float)
    y_pred = df["charge_pred"].values.astype(float)

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)

    # Clasificar átomos
    is_new     = df["atomtype_real"].isin(NEW_ATOMTYPES)
    is_known   = ~is_new

    fig, ax = plt.subplots(figsize=(8, 7))

    # Átomos conocidos
    ax.scatter(
        y_true[is_known], y_pred[is_known],
        alpha=0.7, color="#378ADD", edgecolors="white",
        linewidth=0.3, s=60, label="Atomtype conocido", zorder=2
    )

    # Átomos nuevos — con etiqueta del atomtype
    for _, row in df[is_new].iterrows():
        ax.scatter(
            row["charge_real"], row["charge_pred"],
            alpha=0.9, color="#D85A30", edgecolors="white",
            linewidth=0.3, s=80, zorder=3
        )
        ax.annotate(
            row["atomtype_real"],
            xy=(row["charge_real"], row["charge_pred"]),
            xytext=(5, 5), textcoords="offset points",
            fontsize=7, color="#D85A30", fontweight="bold"
        )

    # Línea de predicción perfecta
    lims = [min(y_true.min(), y_pred.min()) - 0.05,
            max(y_true.max(), y_pred.max()) + 0.05]
    ax.plot(lims, lims, "r--", linewidth=1.2, label="Predicción perfecta", zorder=1)

    ax.set_xlabel("Charge real (e)", fontsize=12)
    ax.set_ylabel("Charge predicha (e)", fontsize=12)
    ax.set_title(f"{title}\nMAE={mae:.4f} | RMSE={rmse:.4f} | R²={r2:.4f}", fontsize=11)

    # Leyenda
    patch_known = mpatches.Patch(color="#378ADD", label="Atomtype conocido")
    patch_new   = mpatches.Patch(color="#D85A30", label="Atomtype nuevo (no visto)")
    ax.legend(handles=[patch_known, patch_new], fontsize=9)

    # MAE separado por grupo
    if is_known.sum() > 0:
        mae_known = mean_absolute_error(y_true[is_known], y_pred[is_known])
        ax.text(0.02, 0.97, f"MAE conocidos: {mae_known:.4f} e",
                transform=ax.transAxes, fontsize=8,
                verticalalignment='top', color="#378ADD")
    if is_new.sum() > 0:
        mae_new = mean_absolute_error(y_true[is_new], y_pred[is_new])
        ax.text(0.02, 0.92, f"MAE nuevos: {mae_new:.4f} e",
                transform=ax.transAxes, fontsize=8,
                verticalalignment='top', color="#D85A30")

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  → {out_path.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True,
                        help="Carpeta con los CSV de resultados")
    parser.add_argument("--output_dir",  required=True,
                        help="Carpeta de salida para las figuras")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(results_dir.glob("results_*.csv"))
    if not csv_files:
        print("No se encontraron CSVs de resultados.")
        return

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        print(f"\n✔ Procesando: {csv_path.name}")
        print(f"  Átomos: {len(df)}")
        print(f"  Atomtypes nuevos encontrados: "
              f"{sorted(df[df['atomtype_real'].isin(NEW_ATOMTYPES)]['atomtype_real'].unique())}")

        # Nombre de la figura
        mol_scheme = csv_path.stem.replace("results_", "")
        mol_name, scheme = mol_scheme.rsplit("_", 1)

        reg_model = "TabTransformer" if scheme == "AA" else "Random Forest"

        plot_scatter_colored(
            df,
            title=f"Charge real vs predicha — {mol_name} {scheme} ({reg_model})",
            out_path=output_dir / f"fig_scatter_colored_{mol_scheme}.png",
            scheme=scheme,
        )

    print(f"\n✔ Figuras guardadas en: {output_dir}")


if __name__ == "__main__":
    main()