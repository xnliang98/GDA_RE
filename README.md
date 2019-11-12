# GDA_RE
tmp readme file


## Preparation
The code requires that you have access to the TACRED dataset (LDC license required). Once you have the TACRED data, please put the JSON files under the directory `dataset/tacred`.

First, download and unzip GloVe vectors:
```shell
chmod +x download.sh
./download.sh
```

Then prepare vocabulary and initial word vectors with:
```shell
python prepare_vocab.py dataset/tacred dataset/vocab --glove_dir dataset/glove
```
This will write vocabulary and word vectors as a numpy matrix into the dir `dataset/vocab`.