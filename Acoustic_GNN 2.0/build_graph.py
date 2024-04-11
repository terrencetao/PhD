import pickle
import os
import pandas as pd
import numpy as np
import tqdm
import networkx as nx
from dgl.data.utils import save_graphs
import argparse 

def calculate_distance_matrix(features):
    # Calculate pairwise distances between features
    dist_matrix = np.linalg.norm(features[:, np.newaxis] - features, axis=2)
    return dist_matrix

def preprocess_data(data):
    # Preprocess data to create a dictionary with ids as keys and features as values
    id_feature_dict = {row['ids']: row['mfcc_feautres'] for _, row in data.iterrows()}
    features = np.array(list(id_feature_dict.values()))
    return id_feature_dict, features

def build_graph(data, threshold, alpha):
    G = nx.Graph()
    id_feature_dict, features = preprocess_data(data)
    dist_matrix = calculate_distance_matrix(features)
    
    for i, (index, row) in tqdm.tqdm(enumerate(data.iterrows()), total=len(data)):
        G.add_node(row['ids'], features=row['mfcc_feautres'])

    for i, (index1, row1) in tqdm.tqdm(enumerate(data.iterrows()), total=len(data)):
        feature1 = row1['mfcc_feautres']
        for j, (index2, row2) in enumerate(data.iterrows()):
            feature2 = row2['mfcc_feautres']
            if index1 != index2:
                distance = dist_matrix[i][j]
                if distance < threshold:
                    if row1['label'] == row2['label']:
                        distance = np.abs(distance / alpha)
                    weight = 1 / distance
                    G.add_edge(row1['ids'], row2['ids'], weight=weight)
    return G
    
def add_node_labels(G, data):
    # Add node labels to the graph as node features
    for node_id, label in data[['ids', 'label']].values:
        G.nodes[node_id].data['label'] = label
    return G


parser = argparse.ArgumentParser()
parser.add_argument('--input_folder', help ='source folder')
   
    #parser.add_argument('--output_folder', help ='output folder')
args = parser.parse_args()
    
input_folder = args.input_folder

threshold = 1000
alpha = 20

with open(os.path.join(input_folder, 'dataset.pl'), 'rb') as file:
    data = pickle.load(file)

G = build_graph(data, threshold, alpha)




# Build the graph
G = build_graph(data, threshold, alpha)

# Add node labels as node features
G = add_node_labels(G, data)

# Convert NetworkX graph to DGL graph
dgl_G = dgl.from_networkx(G, node_attrs=['features', 'label'])

graph_labels = {"glabel": torch.tensor([0])}
save_graphs(os.path.join(input_folder,"origin_word_graph_data.bin"),[dgl_graph] , graph_labels)

