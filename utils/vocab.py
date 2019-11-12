"""
A class for basic vocab operations.
"""

import os
import random
import numpy as np
import pickle

from utils import constant

# set random seed
random_seed = 1234
random.seed(random_seed)
np.random.seed(random_seed)

def build_embedding(wv_file, vocab, wv_dim):
    ''' Get embedding of vocab.
        Args:
            wv_file (str): word vectors filename and path.
            vocab (list): word list of vocab.
            wv_dim (int): dimension of word embedding.
        returns:
            emb (np.array): embedding matrix for vocab
    '''
    vocab_size = len(vocab)
    # random initialize emb matrix with uniform distribution (-1, 1)
    emb = np.random.uniform(-1, 1, (vocab_size, wv_dim))
    emb[constant.PAD_ID] = 0 # <PAD> 's embedding should be zero

    w2id = {w: i for i, w in enumerate(vocab)}
    with open(wv_file, encoding='utf-8') as fin:
        for line in fin:
            elems = line.split()
            token = "".join(elems[: -wv_dim])
            if token in w2id:
                emb[w2id[token]] = [float(v) for v in elems[-wv_dim: ]]
    return emb

def load_glove_vocab(file, wv_dim):
    ''' Load vocab of glove file
        Args:
            file (str): glove file path.
            wv_dim (int): word vector dimension.
        returns:
            vocab (set): vocab of glove.
    '''
    vocab = set()
    with open(file, encoding='utf-8') as fin:
        for line in fin:
            elems = line.split()
            token = "".join(elems[0: -wv_dim])
            vocab.add(token)
    return vocab

def Vocab(object):
    """ Class of Vocabulary. 
        Attrs:
            id2word dict, id-word map
            word2id dict, word-id map
            size int, Vocab size
            embeddings np.array, embedding matrix
            if load = False:
                word_counter Counter: Counter used to create vocab
    """
    def __init__(self, filename, load=False, word_counter=None, freq=0):
        ''' Create vocab or load vocab with filename and word_counter.
            Args:
                filename (str): filename used to save or load vocab.
                load (boolean): if true, load vocab from filename.
                word_counter (collections.Counter): Counter used to create
                    vocab.
                freq (int): if word frequence > freq, keep it.
        '''
        if load:
            assert os.path.exists(filename), "Vocab file does not exist " +\
                "at " + filename
            self.id2word, self.word2id = self.load(filename)
            self.size = len(self.id2word)
            print("Vocab size {} loaded from file {}.".format(self.size, filename))
        else:
            print("Creating vocab from scratch ...")
            assert word_counter is not None, "word counter is not provided for vocab creation."
            self.word_counter = word_counter
            if freq > 1:
                # remove words that occur less than freq
                self.word_counter = {k: v for k, v in self.word_counter.items() \
                    if v >= freq}
            self.id2word = sorted(word_counter, key=lambda k: self.word_counter[k], reverse=True)
            # add <PAD> and <UNK> to id2word in head
            self.id2word =  [constant.PAD_TOKEN, constant.UNK_TOKEN] + self.id2word
            self.word2id = {v: k for k, v in self.id2word.items()}
            self.size = len(self.id2word)
            self.save(filename)
            print("Vocab size {} saved to file {}.".format(self.size, filename))

    def load(self, filename):
        """ Load vocab's word2id and id2word from filename.
            Args:
                filename (str): path of saved vocab.
            returns:
                id2word (dict): id: word map
                word2id (dict): word: id map
        """
        with open(filename, 'rb') as fin:
            id2word = pickle.load(fin)
            word2id = {v: k for k, v in id2word.items()}
        return id2word, word2id

    def save(self, filename):
        """ Save id2word as vocab to filename, save as pickle file.
            Args:
                filename (str): path to save id2word.
        """
        if os.path.exists(filename):
            print("Overwriting old vocab file at " + filename)
            os.remove(filename)
        with open(filename, 'wb') as fout:
            pickle.dump(self.id2word, fout)
        return None
        
    def map(self, token_list):
        ''' Map token list to id list.
            Args:
                token_list (list): token list.
            returns:
                id list of tokens
        '''
        return [self.word2id[w] if w in self.word2id else constant.UNK_ID \
            for w in token_list]

    def unmap(self, idx_list):
        ''' Map id list to token list
            Args:
                idx_list (list): idx list.
            returns:
                token list of idx
        '''
        return [self.id2word[x] for x in idx_list]

    def get_embedding(self, word_vector=None, dim=100):
        ''' Get embedding matrix of vocab.
            word_vector (dict): word vector index by word.
            dim (int): dimension of word embedding.
        '''
        # init of embeddings
        self.embeddings = 2 * constant.EMB_INIT_RANGE * np.random.rand(self.size, dim) - constant.EMB_INIT_RANGE
        if word_vector is not None:
            assert len(list(word_vector.values())[0]) == dim, \
                "Word vectors does not have requires dimension {}".format(dim)
            for w, idx in self.word2id.items():
                if w in word_vector:
                    self.embeddings[idx] = np.asarray(word_vector[w])
        return self.embeddings