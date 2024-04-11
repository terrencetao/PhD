import dgl
import torch
import itertools
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import argparse 
from data_processing import load_pickle, save_pickle
import os
from dgl.data.utils import save_graphs

def build_co_occurrence_matrix(sentences, vocab):
    # Créer une matrice de co-occurrence initialisée à zéro
    co_occurrence_matrix = np.zeros((len(vocab), len(vocab)), dtype=int)
    # Créer un dictionnaire pour mapper les mots du vocabulaire à leurs indices dans la matrice
    word_to_index = {word: i for i, word in enumerate(vocab)}
    
    # Parcourir chaque phrase dans l'ensemble de phrases
    for sentence in sentences:
        # Créer des paires de mots uniques dans chaque phrase
        unique_words = set(sentence)
        # Générer toutes les combinaisons de paires de mots dans la phrase
        word_combinations = itertools.combinations(unique_words, 2)
        # Mettre à jour la matrice de co-occurrence pour chaque paire de mots
        for word_pair in word_combinations:
            if word_pair[0] in vocab and word_pair[1] in vocab:
                index1 = word_to_index[word_pair[0]]
                index2 = word_to_index[word_pair[1]]
                co_occurrence_matrix[index1][index2] += 1
                co_occurrence_matrix[index2][index1] += 1  # La matrice est symétrique
    
    return co_occurrence_matrix

def build_graph(co_occurrence_matrix, vocab):
    G = nx.Graph()

    for i, word in enumerate(vocab):
        G.add_node(word)

    for i in range(len(vocab)):
        for j in range(i + 1, len(vocab)):
            weight = co_occurrence_matrix[i, j]
            if weight > 0:
                G.add_edge(vocab[i], vocab[j], weight=weight)

    return G



# Obtenir la représentation one-hot pour les nœuds
def one_hot(node_id, num_nodes):
    x = torch.zeros(num_nodes)
    x[node_id] = 1
    return x
    
parser = argparse.ArgumentParser()
parser.add_argument('--input_folder', help ='source folder')

args = parser.parse_args()
    
input_folder = args.input_folder



# Exemple d'utilisation
corpus = load_pickle(file_path=os.path.join(input_folder,'sentences.pkl'))
vocab = load_pickle(file_path=os.path.join(input_folder,'vocabulary.pkl'))
window_size = 2

co_occurrence_matrix = build_co_occurrence_matrix(corpus, vocab)
graph = build_graph(co_occurrence_matrix, vocab)



# Construire le graphe DGL
dgl_graph = dgl.DGLGraph(graph)
 
# Attribuer les représentations one-hot aux nœuds
num_nodes = len(vocab)
node_features = torch.stack([one_hot(i, num_nodes) for i in range(num_nodes)])

# Ajouter les features aux noeuds du graphe
dgl_graph.ndata['feat'] = node_features

graph_labels = {"glabel": torch.tensor([0])}
save_graphs(os.path.join(input_folder,"origin_word_graph_data.bin"),[dgl_graph] , graph_labels)



