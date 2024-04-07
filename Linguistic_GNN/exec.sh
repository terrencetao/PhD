for i in $(seq 100 100 1500); do

  rm origin_word_graph_data.bin partial_graph.pkl sentences.pkl 
  python3 data_processing.py --input_folder '' --number_sentence $i

  python3 word2vec_model.py --input_folder ''

  python3 build_graph.py --input_folder ''

  python3 gnn_model.py --input_folder ''

  python3 w2v_vs_gnn.py --input_folder '' --model 'w2v'

  python3 w2v_vs_gnn.py --input_folder '' --model 'gnn'
done


