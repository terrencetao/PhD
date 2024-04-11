import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import SAGEConv
from torch_geometric.nn import global_mean_pool
import numpy as np
from data_processing import load_pickle, save_pickle
import os
import argparse 
from dgl.data.utils import load_graphs


# Définir le modèle GCN avec SageConv
class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats):
        super(GCN, self).__init__()
        self.conv1 = SAGEConv(in_feats, hidden_size, 'mean')
        self.conv2 = SAGEConv(hidden_size, out_feats, 'mean')

    def forward(self, g, features):
        x = F.relu(self.conv1(g, features))
        x = self.conv2(g, x)
        return x



# Fonction d'entraînement
def train(model, graph, features, labels, epochs):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(graph, features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')




def generate_adjacency_matrix(graph, vocab):
    adj_matrix = np.zeros((len(vocab), len(vocab)), dtype=np.float32)
    src_ids, dst_ids = graph.edges()
    src_ids = src_ids.numpy()
    dst_ids = dst_ids.numpy()
    for src_id, dst_id in zip(src_ids, dst_ids):
        src = vocab[src_id]
        dst = vocab[dst_id]
        src_index = vocab.index(src)
        dst_index = vocab.index(dst)
        adj_matrix[src_index, dst_index] = 1
        adj_matrix[dst_index, src_index] = 1  # Si votre graphe est non-dirigé, décommentez cette ligne
    return adj_matrix

    
    
    
parser = argparse.ArgumentParser()
parser.add_argument('--input_folder', help ='source folder')
   
    #parser.add_argument('--output_folder', help ='output folder')
args = parser.parse_args()
    
input_folder = args.input_folder




glist, label_dict = load_graphs(os.path.join(input_folder,"origin_word_graph_data.bin"))
dgl_graph = glist[0]

# Initialiser le modèle GCN
vocab = load_pickle(file_path=os.path.join(input_folder,'vocabulary.pkl'))
input_size = len(vocab)  # Taille d'entrée = nombre de mots dans le vocabulaire
hidden_size = 64
output_size = 128
model = GCN(input_size, hidden_size, output_size)

# Définir la fonction de perte et l'optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Créer la matrice d'adjacence binaire
adj_matrix = generate_adjacency_matrix(dgl_graph, vocab)

# Utiliser la matrice d'adjacence comme labels pour les nœuds
node_labels = torch.tensor(adj_matrix)
# Convertir la matrice d'adjacence en un tenseur one-hot encoding
num_nodes = len(vocab)
one_hot_labels = torch.zeros(num_nodes, dtype=torch.long)

# Entraîner le modèle avec les étiquettes reformulées
train(model, dgl_graph, dgl_graph.ndata['feat'], one_hot_labels, epochs=100)



# Récupérer les représentations des nœuds en sortie
with torch.no_grad():
    model.eval()
    node_representations = model(dgl_graph, dgl_graph.ndata['feat'])

word_to_vec={}
# Afficher les représentations des nœuds en sortie et leur mot correspondant
for i, representation in enumerate(node_representations):
    word = list(vocab)[i]
    word_to_vec[word] = representation
    
   
save_pickle(data=word_to_vec , file_path=os.path.join(input_folder,'word2vec_gnn.pkl'))
    
