import pickle
from pathlib import Path

# Ruta al archivo generado
FILE_PATH = "dataset_grafos.pkl"

def inspeccionar_grafos():
    if not Path(FILE_PATH).exists():
        print(f"❌ Error: No se encuentra el archivo {FILE_PATH}")
        return

    # Cargar el archivo binario
    with open(FILE_PATH, "rb") as f:
        data = pickle.load(f)

    print(f"✅ Archivo cargado correctamente.")
    print(f"📊 Total de moléculas en el dataset: {len(data)}")
    
    # Listar las primeras 5 moléculas
    moleculas = list(data.keys())
    print(f"🧬 Primeras moléculas: {moleculas[:5]}")

    # Ver el detalle de la primera molécula
    if moleculas:
        primera_mol = moleculas[0]
        print(f"\n🔍 Detalle de la molécula: {primera_mol}")
        print(f"   - Cantidad de átomos (nodos): {len(data[primera_mol]['nodos'])}")
        print(f"   - Cantidad de enlaces (aristas): {len(data[primera_mol]['aristas'])}")
        
        # Mostrar el primer átomo como ejemplo
        primer_id = list(data[primera_mol]['nodos'].keys())[0]
        print(f"   - Ejemplo Nodo {primer_id}:", data[primera_mol]['nodos'][primer_id])

if __name__ == "__main__":
    inspeccionar_grafos()
