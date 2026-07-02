"""
analisis_distribucion.py
------------------------
Genera figuras de distribución de clases de atomtype
antes y después del remapeo, para incluir en la tesis.

Uso:
    python analisis_distribucion.py \
        --before dataset.csv \
        --after dataset_remapped.csv \
        --output_dir figuras_distribucion/
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"]  = 11


def plot_distribucion(df, title, output_path, color):
    counts = df["atomtype"].value_counts().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(counts.index, counts.values, color=color, edgecolor="white")
    ax.set_xlabel("Atomtype")
    ax.set_ylabel("Cantidad de átomos")
    ax.set_title(title)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, rotation=45, ha="right")
    # Agregar valores encima de cada barra
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                str(val), ha="center", va="bottom", fontsize=9)
    ax.axhline(y=counts.mean(), color="red", linestyle="--",
               linewidth=1, label=f"Media: {counts.mean():.1f}")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {output_path.name}")


def plot_comparacion(df_before, df_after, output_path):
    """Figura comparativa antes/después del remapeo."""
    clases_antes  = df_before["atomtype"].value_counts()
    clases_despues = df_after["atomtype"].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for ax, counts, title, color in [
        (axes[0], clases_antes,  f"Antes del remapeo ({len(clases_antes)} clases)",  "#378ADD"),
        (axes[1], clases_despues, f"Después del remapeo ({len(clases_despues)} clases)", "#D85A30"),
    ]:
        bars = ax.bar(counts.index, counts.values, color=color, edgecolor="white")
        ax.set_xlabel("Atomtype")
        ax.set_ylabel("Cantidad de átomos")
        ax.set_title(title)
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(counts.index, rotation=45, ha="right")
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    str(val), ha="center", va="bottom", fontsize=8)

    plt.suptitle("Distribución de clases de atomtype — antes y después del remapeo",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {output_path.name}")


def imprimir_resumen(df_before, df_after):
    print("\n=== Clases antes del remapeo ===")
    print(df_before["atomtype"].value_counts().to_string())
    print(f"\nTotal clases: {df_before['atomtype'].nunique()}")
    print(f"Total átomos: {len(df_before)}")

    print("\n=== Clases después del remapeo ===")
    print(df_after["atomtype"].value_counts().to_string())
    print(f"\nTotal clases: {df_after['atomtype'].nunique()}")
    print(f"Total átomos: {len(df_after)}")

    # Clases que desaparecieron
    clases_antes  = set(df_before["atomtype"].unique())
    clases_despues = set(df_after["atomtype"].unique())
    fusionadas = clases_antes - clases_despues
    if fusionadas:
        print(f"\nClases fusionadas: {sorted(fusionadas)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before",     required=True, help="dataset.csv (antes del remapeo)")
    parser.add_argument("--after",      required=True, help="dataset_remapped.csv (después)")
    parser.add_argument("--output_dir", default="figuras_distribucion/")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_before = pd.read_csv(args.before)
    df_after  = pd.read_csv(args.after)

    imprimir_resumen(df_before, df_after)

    print("\nGenerando figuras...")

    plot_distribucion(
        df_before,
        f"Distribución de atomtypes antes del remapeo ({df_before['atomtype'].nunique()} clases)",
        out_dir / "fig_distribucion_antes.png",
        color="#378ADD"
    )

    plot_distribucion(
        df_after,
        f"Distribución de atomtypes después del remapeo ({df_after['atomtype'].nunique()} clases)",
        out_dir / "fig_distribucion_despues.png",
        color="#D85A30"
    )

    plot_comparacion(
        df_before, df_after,
        out_dir / "fig_comparacion_remapeo.png"
    )

    print(f"\n✔ Figuras guardadas en: {out_dir}")


if __name__ == "__main__":
    main()