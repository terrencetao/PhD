import pickle
import os
import pandas as pd
import numpy as np
import tqdm
import networkx as nx
from dgl.data.utils import save_graphs
import argparse 
import dgl
import torch
import random

def calculate_distance_matrix(features):
    # Calculate pairwise distances between features
    dist_matrix = np.linalg.norm(features[:, np.newaxis] - features, axis=2)
    return dist_matrix

def build_graph(data, threshold, alpha):
    G = nx.Graph()
    id_feature_dict, features = preprocess_data(data)
    dist_matrix = calculate_distance_matrix(features)
    
    for i, (index, row) in enumerate(data.iterrows()):
        G.add_node(row['ids'], features=row['mfcc_feautres'])

    for i, (index1, row1) in tqdm.tqdm(enumerate(data.iterrows())):
        for j, (index2, row2) in enumerate(data.iterrows()):
            if index1 != index2:
                distance = dist_matrix[i][j]
                if np.isscalar(distance) and distance < threshold:
                    if row1['label'] == row2['label']:
                        distance = np.abs(distance / alpha)
                    weight = 1 / distance
                    G.add_edge(row1['ids'], row2['ids'], weight=weight)
                elif np.any(distance < threshold):
                    indices = np.where(distance < threshold)[0]
                    for idx in indices:
                        if idx < len(data) and row1['label'] == data.iloc[idx]['label']:
                            distance_at_idx = np.abs(distance[idx] / alpha)
                            weight = 1 / distance_at_idx
                            G.add_edge(row1['ids'], data.iloc[idx]['ids'], weight=weight)
    return G

def preprocess_data(data):
    id_feature_dict = {row['ids']: row['mfcc_feautres'] for _, row in data.iterrows()}
    features = np.array(list(id_feature_dict.values()))
    return id_feature_dict, features

def add_node_labels(G, data):
    label_dict = {row['ids']: row['label'] for _, row in data.iterrows()}
    nx.set_node_attributes(G, label_dict, 'label')
    return G

def convert_to_dgl_graph(G):
    dgl_G = dgl.from_networkx(G, node_attrs=['features'])
    return dgl_G

parser = argparse.ArgumentParser()
parser.add_argument('--input_folder', help ='source folder')
   
    #parser.add_argument('--output_folder', help ='output folder')
args = parser.parse_args()
    
input_folder = args.input_folder

threshold = 1000
alpha = 20

with open(os.path.join(input_folder, 'dataset.pl'), 'rb') as file:
    datcdo

# Build the graph
G = build_graph(data, threshold, alpha)

# Add node labels as node attributes
G = add_node_labels(G, data)

# Convert NetworkX graph to DGL graph
dgl_graph = convert_to_dgl_graph(G)

graph_labels = {"glabel": torch.tensor([0])}
save_graphs(os.path.join(input_folder,"origin_word_graph_data.bin"),[dgl_graph] , graph_labels)

