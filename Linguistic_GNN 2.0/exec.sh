rm resultatsgnn resultatsw2v

for i in $(seq 1000 500 5000); do

  python3 data_processing.py --input_folder '' --number_sentence $i

  python3 word2vec_model.py --input_folder ''

  python3 build_graph.py --input_folder ''

  python3 gnn_model.py --input_folder ''

  python3 evaluation.py --input_folder '' 
done


