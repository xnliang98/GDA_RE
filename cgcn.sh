if [ ! -d "dataset/vocab" ]; then
  python prepare_vocab.py dataset/tacred dataset/vocab
fi

python train_cgcn.py --id 00 --seed 1234 --prune_k 1 --lr 0.3 --hidden_dim 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --info "CGCN model" 

python eval.py saved_models/00 > res.log