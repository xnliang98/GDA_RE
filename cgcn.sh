if [ ! -d "dataset/vocab" ]; then
  python prepare_vocab.py dataset/tacred dataset/vocab
fi

python train_cgcn.py --id 00 --lr 0.3 --conv_l2 0.002 --pooling_l2 0.002 --info "CGCN model" 

python eval.py > res.log