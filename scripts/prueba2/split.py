import argparse
import pandas as pd


def split_AA_UA(input_path, output_AA, output_UA):
    # ─────────────────────────────────────────────
    # 1. Cargar dataset
    # ─────────────────────────────────────────────
    df = pd.read_csv(input_path)

    print(f"\n📥 Dataset original: {df.shape}")

    # ─────────────────────────────────────────────
    # 2. DATASET AA (sin cambios)
    # ─────────────────────────────────────────────
    df_AA = df.copy()

    # ─────────────────────────────────────────────
    # 3. DATASET UA
    # ─────────────────────────────────────────────
    # Regla:
    # - eliminar H ligados a C
    # - conservar H ligados a N u O

    def is_polar_h(row):
        if row["element"] != "H":
            return True  # no es H → se queda

        # si tiene vecinos N u O → es polar → se queda
        if row["N_count"] > 0 or row["O_count"] > 0:
            return True

        # si no → H unido a C → se elimina
        return False

    df_UA = df[df.apply(is_polar_h, axis=1)].copy()

    # ─────────────────────────────────────────────
    # 4. Guardar
    # ─────────────────────────────────────────────
    df_AA.to_csv(output_AA, index=False)
    df_UA.to_csv(output_UA, index=False)

    # ─────────────────────────────────────────────
    # 5. Info
    # ─────────────────────────────────────────────
    print(f"\n📊 AA: {df_AA.shape}")
    print(f"📊 UA: {df_UA.shape}")
    print(f"🗑️ Eliminados (UA): {len(df_AA) - len(df_UA)} átomos")

    print(f"\n💾 Guardados:")
    print(f" - {output_AA}")
    print(f" - {output_UA}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Divide dataset en AA y UA")

    parser.add_argument("--input", "-i", required=True, help="Dataset original")
    parser.add_argument("--output_AA", required=True, help="Salida All-Atom")
    parser.add_argument("--output_UA", required=True, help="Salida United-Atom")

    args = parser.parse_args()

    split_AA_UA(
        input_path=args.input,
        output_AA=args.output_AA,
        output_UA=args.output_UA
    )