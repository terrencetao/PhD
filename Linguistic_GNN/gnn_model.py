import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import numpy as np
import scipy.sparse as sp
import os


import dgl.data
from dgl.data.utils import load_graphs
import pickle
from dgl.nn import SAGEConv
from data_processing import load_pickle, save_pickle
from sklearn.metrics import roc_auc_score
import argparse 




# ----------- 2. create model -------------- #
# build a two-layer GraphSAGE model
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, 'mean')
        self.conv2 = SAGEConv(h_feats, h_feats, 'mean')
        self.conv3 = SAGEConv(h_feats, h_feats, 'mean')

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        h = F.relu(h)
        h = self.conv3(g, h)
        return h

import dgl.function as fn

class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return g.edata['score'][:, 0]


class MLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, 1)

    def apply_edges(self, edges):
        """
        Computes a scalar score for each edge of the given graph.

        Parameters
        ----------
        edges :
            Has three members ``src``, ``dst`` and ``data``, each of
            which is a dictionary representing the features of the
            source nodes, the destination nodes, and the edges
            themselves.

        Returns
        -------
        dict
            A dictionary of new edge features.
        """
        h = torch.cat([edges.src['h'], edges.dst['h']], 1)
        return {'score': self.W2(F.relu(self.W1(h))).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['score']

def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    return F.binary_cross_entropy_with_logits(scores, labels)

def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return roc_auc_score(labels, scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', help ='source folder')
   
    #parser.add_argument('--output_folder', help ='output folder')
    args = parser.parse_args()
    
    input_folder = args.input_folder




    glist, label_dict = load_graphs(os.path.join(input_folder,"origin_word_graph_data.bin"))
    g = glist[0]

# Split edge set for training and testing
    u, v = g.edges()

    eids = np.arange(g.number_of_edges())
    eids = np.random.permutation(eids)
    test_size = int(len(eids) * 0)
    train_size = g.number_of_edges() - test_size
# test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]

# Find all negative edges and split them for training and testing
    adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
    adj_neg = 1 - adj.todense() - np.eye(g.number_of_nodes())
    neg_u, neg_v = np.where(adj_neg != 0)

    neg_eids = np.random.choice(len(neg_u), g.number_of_edges())
#test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
    train_neg_u, train_neg_v = neg_u[neg_eids[test_size:]], neg_v[neg_eids[test_size:]]

    train_g = dgl.remove_edges(g, eids[:test_size])



    train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.number_of_nodes())
    train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.number_of_nodes())




    model = GraphSAGE(train_g.ndata['feat'].shape[1], 50)
# You can replace DotPredictor with MLPPredictor.
    pred = MLPPredictor(50)
# pred = DotPredictor()



#----------- 3. set up loss and optimizer -------------- #
# in this case, loss will in training loop
    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=0.01)

# ----------- 4. training -------------------------------- #
    all_logits = []
    for e in range(30):
    # forward
       h = model(train_g, train_g.ndata['feat'])
       pos_score = pred(train_pos_g, h)
       neg_score = pred(train_neg_g, h)
       loss = compute_loss(pos_score, neg_score)

    # backward
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()

       if e % 5 == 0:
          print('In epoch {}, loss: {}'.format(e, loss))

# ----------- 5. check results ------------------------ #

    with torch.no_grad():
       pos_score = pred(train_pos_g, h)
       neg_score = pred(train_neg_g, h)
       print('AUC', compute_auc(pos_score, neg_score))
     
    
    
    
    
    
    

    word_to_vec={}
    node_features = h.detach().numpy()

    word_to_id = load_pickle(file_path=os.path.join(input_folder,'word_to_id.pkl'))
    words = load_pickle(file_path=os.path.join(input_folder,'vocabulary.pkl'))
   
    
    
    for word, id_ in word_to_id.items():
       word_to_vec[word] = node_features[id_]



    save_pickle(data=word_to_vec , file_path=os.path.join(input_folder,'word2vec_gnn.pkl'))



