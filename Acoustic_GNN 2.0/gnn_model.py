import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SAGEConv
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from dgl.data.utils import load_graphs
import argparse 

# Define the CNN model
class CNN(nn.Module):
    def __init__(self, in_channels, conv_param):
        super(CNN, self).__init__()
        layers = []
        in_channels = in_channels
        for i in range(len(conv_param) - 1):
            layers.append(nn.Conv1d(in_channels=conv_param[i], out_channels=conv_param[i+1], kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            in_channels = conv_param[i+1]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)

# Define the GCN model
class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes, conv_param):
        super(GCN, self).__init__()
        self.cnn = CNN(in_channels=conv_param[0], conv_param=conv_param)
        self.conv1 = SAGEConv(conv_param[-1], hidden_size, 'mean')
        self.conv2 = SAGEConv(hidden_size, num_classes, 'mean')

    def forward(self, g, features):
        x = self.cnn(features.unsqueeze(1)).squeeze(1)
        x = F.relu(self.conv1(g, x))
        x = self.conv2(g, x)
        return x

# Define the training function
def train(model, g, features, labels, epochs=100, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        logits = model(g, features)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')
            


# Define a custom topological loss function
def topological_loss(embeddings, adj_matrix):
    # Calculate pairwise cosine similarity between embeddings
    cosine_sim = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
    
    # Compute the reconstruction loss
    reconstruction_loss = F.mse_loss(cosine_sim, adj_matrix)
    
    return reconstruction_loss

# Define the training function with topological loss
def train_with_topological_loss(model, g, features, adj_matrix, epochs=100, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        embeddings = model(g, features)
        loss = topological_loss(embeddings, adj_matrix)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')




parser = argparse.ArgumentParser()
parser.add_argument('--input_folder', help ='source folder')
   
#parser.add_argument('--output_folder', help ='output folder')
args = parser.parse_args()
    
input_folder = args.input_folder



glist, label_dict = load_graphs(os.path.join(input_folder,"origin_word_graph_data.bin"))
dgl_graph = glist[0]  

# Initialize the GCN model
in_feats = dgl_G.ndata['features'].shape[1]
hidden_size = 64
num_classes = len(data['label'].unique())  # Number of unique labels
conv_param = [1, 3, 32]  # CNN parameters: [input_channels, hidden_channels, output_channels]
model = GCN(in_feats, hidden_size, num_classes, conv_param)



# Train the model
features = dgl_G.ndata['features']
labels = torch.tensor(data['label'].values)
train(model, dgl_G, features, labels)

# Define the file path for saving the model
model_path = os.path.join(input_folder,"gnn_model.pth")

# Save the model
torch.save(model.state_dict(), model_path)






# Train the model with topological loss
# Assume adj_matrix is the adjacency matrix of the graph
adj_matrix = torch.tensor(nx.to_numpy_matrix(G))
train_with_topological_loss(model, dgl_G, features, adj_matrix)
model_path_sup = os.path.join(input_folder,"gnn_model_unsup.pth")
