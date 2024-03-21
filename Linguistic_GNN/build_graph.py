import dgl
import torch
import networkx as nx
import numpy as np
import tqdm
import pickle
from dgl.data.utils import save_graphs
import torch
import argparse 
from data_processing import load_pickle, save_pickle
import os


class WordContextDataset(dgl.data.DGLDataset):
    def __init__(self, sentences,  vocab, context_window_size=2):
        super(WordContextDataset, self).__init__(name='word_context')
        self.sentences = sentences
        self.context_window_size = context_window_size
        self.vocabulary = vocab
        self._prepare()

    def _prepare(self):
        # Create a directed graph using NetworkX
        graph = nx.DiGraph()

        # Create word-to-ID and ID-to-word mappings
        word_to_id = {}
        id_to_word = {}

        # Function to add a word to the mappings
        def add_word(word):
            if word not in word_to_id:
                new_id = len(word_to_id)
                word_to_id[word] = new_id
                id_to_word[new_id] = word

        # Iterate through each tokenized sentence
        # tokenized_sentences = [word_tokenize(sentence) for sentence in self.sentences]
        for sentence_tokens in tqdm.tqdm(self.sentences):
            # Iterate through each word in the sentence
            for i, word in enumerate(sentence_tokens):
                if word in self.vocabulary:
                # Add the central word to the mappings
                  add_word(word)

                  # Define the start and end indices of the context window
                  start_index = max(0, i - self.context_window_size)
                  end_index = min(len(sentence_tokens), i + self.context_window_size + 1)

                  # Extract the context words within the window
                  context_words = sentence_tokens[start_index:i] + sentence_tokens[i+1:end_index]

                  # Add edges to the graph based on the context
                  for context_word in context_words:
                    if context_word in self.vocabulary:
                      add_word(context_word)
                      #graph.add_edge(word_to_id[word], word_to_id[context_word])
                      graph.add_edge(word, context_word)

        # Convert the NetworkX graph to a DGL graph
        self.graph = dgl.DGLGraph(graph)

        # Store word-to-ID and ID-to-word mappings as attributes of the dataset
        self.word_to_id = word_to_id
        self.id_to_word = id_to_word
        self.word_to_node = {word:node_index for node_index, word in enumerate(graph.nodes)}
        
    def __getitem__(self, idx):
        return self.graph

    def __len__(self):
        return 1  # This dataset consists of a single graph

# Sample sentences
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', help ='source folder')
    #parser.add_argument('--output_folder', help ='output folder')
    args = parser.parse_args()
    
    input_folder = args.input_folder
    
    if not os.path.isfile(os.path.join(input_folder,"partial_graph.pkl")):
# Create the WordContextDataset
        print('Create the WordContextDataset')
        words = load_pickle(file_path=os.path.join(input_folder,'vocabulary.pkl'))
        
        sentences = load_pickle(file_path=os.path.join(input_folder,'sentences.pkl'))
        
        dataset = WordContextDataset(sentences,words, context_window_size=7)

        save_pickle(dataset, os.path.join(input_folder,"partial_graph.pkl"))
        
    dataset = load_pickle(os.path.join(input_folder,"partial_graph.pkl"))
    
    try:
        dataset.graph.ndata['feat'] = torch.eye(len(dataset.word_to_id)-1)
    except:
        dataset.graph.ndata['feat'] = torch.eye(len(dataset.word_to_id))
        print('take the except root')
   
    graph_labels = {"glabel": torch.tensor([0])}
    save_graphs(os.path.join(input_folder,"origin_word_graph_data.bin"),[dataset[0]] , graph_labels)

    save_pickle(data=dataset.id_to_word , file_path=os.path.join(input_folder,'id_to_word.pkl'))
    save_pickle(data=dataset.word_to_id , file_path=os.path.join(input_folder,'word_to_id.pkl'))
   
    save_pickle(data=dataset.word_to_node , file_path=os.path.join(input_folder,'word_to_node.pkl'))

