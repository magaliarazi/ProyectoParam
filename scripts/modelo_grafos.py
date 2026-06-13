import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import LabelEncoder
import numpy as np

# 1. Tu dataset (resumido para el ejemplo, pero compatible con tu estructura)
dataset_raw = {
    # Aquí iría el diccionario que pegaste...
    'FFX9': { ... }, 
    # etc...
}

def prepare_data(raw_data):
    graphs = []
    # Recolectar todos los elementos para un encoding consistente
    all_elements = []
    for mol in raw_data.values():
        for node in mol['nodos'].values():
            all_elements.append(node['elemento'])
    
    le = LabelEncoder()
    le.fit(all_elements)
    
    for key, mol in raw_data.items():
        nodos = mol['nodos']
        aristas = mol['aristas']
        
        # --- Nodo Features ---
        node_features = []
        node_targets = []
        node_map = {} # Para mapear id original a id 0-indexed
        
        for i, (idx, attr) in enumerate(nodos.items()):
            # Features: [Elemento_codificado, Masa, Es_planar]
            elem_enc = le.transform([attr['elemento']])[0]
            node_features.append([elem_enc, attr['masa'], attr['es_planar']])
            node_targets.append([attr['carga_target']])
            node_map[idx] = i
            
        x = torch.tensor(node_features, dtype=torch.float)
        y = torch.tensor(node_targets, dtype=torch.float)
        
        # --- Edge Index (Conexiones) ---
        edge_indices = []
        edge_attr = []
        for edge in aristas:
            u, v = edge['indices']
            # PyG usa grafos dirigidos, para moléculas solemos duplicar las aristas
            edge_indices.append([node_map[u], node_map[v]])
            edge_indices.append([node_map[v], node_map[u]])
            edge_attr.append([edge['distancia_A']])
            edge_attr.append([edge['distancia_A']])
            
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(edge_attr, dtype=torch.float)
        
        graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y))
        
    return graphs, le

# 2. Definición del Modelo GNN
class GNNModel(torch.nn.Module):
    def __init__(self, num_node_features):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, 32)
        self.out = torch.nn.Linear(32, 1) # Salida: 1 valor por nodo (la carga)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Capa de convolución 1
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
        # Capa de convolución 2
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Regresión final
        return self.out(x)

# 3. Entrenamiento
# Suponiendo que 'dataset_grafos' es tu variable con el diccionario
graphs, encoder = prepare_data(dataset_raw)
loader = DataLoader(graphs, batch_size=2, shuffle=True)

model = GNNModel(num_node_features=3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

model.train()
for epoch in range(100):
    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss/len(loader):.4f}")

print("Entrenamiento completado.")
