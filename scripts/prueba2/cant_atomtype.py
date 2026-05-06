"python analyze_atomtypes.py -i tu_dataset.csv -o salida_eda"

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_atomtypes(input_path, output_dir):
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # =========================
    # LOAD
    # =========================
    df = pd.read_csv(input_path)

    if "atomtype" not in df.columns:
        raise ValueError("❌ La columna 'atomtype' no existe en el dataset")

    print(f"\n📥 Dataset cargado: {input_path}")
    print(f"📊 Total filas: {len(df)}")

    # =========================
    # COUNT
    # =========================
    counts = df["atomtype"].value_counts()
    pct = (counts / len(df) * 100).round(2)

    summary = pd.DataFrame({
        "count": counts,
        "percentage": pct
    })

    print("\n📊 Distribución de atomtypes:")
    print(summary)

    # =========================
    # RARE CLASSES
    # =========================
    threshold = 10
    rare = counts[counts < threshold]

    print(f"\n⚠ Clases con menos de {threshold} muestras:")
    print(rare)

    # =========================
    # SAVE CSV
    # =========================
    summary.to_csv(output_dir / "atomtype_distribution.csv")

    # =========================
    # PLOT
    # =========================
    plt.figure(figsize=(16, 6))

    sns.barplot(
        x=counts.index,
        y=counts.values
    )

    plt.title("Distribución de atomtypes")
    plt.xlabel("Atomtype")
    plt.ylabel("Cantidad")
    plt.xticks(rotation=90)

    plt.tight_layout()
    plt.savefig(output_dir / "atomtype_distribution.png", dpi=300)
    plt.show()

    # =========================
    # INFO FINAL
    # =========================
    print("\n📁 Archivos generados:")
    print(f"- {output_dir / 'atomtype_distribution.csv'}")
    print(f"- {output_dir / 'atomtype_distribution.png'}")

    print("\n✅ Análisis completo")


# =========================
# CLI
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Análisis de atomtypes en dataset crudo")

    parser.add_argument("--input", "-i", required=True, help="Archivo CSV de entrada")
    parser.add_argument("--output", "-o", default="eda_atomtypes", help="Carpeta de salida")

    args = parser.parse_args()

    analyze_atomtypes(args.input, args.output)