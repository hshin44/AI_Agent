# coding: utf-8
import numpy as np
import cPickle
from collections import defaultdict
import sys, re
import pandas as pd

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--POSITIVE_DATA', type = str)
parser.add_argument('-n', '--NEGATIVE_DATA', type = str)
parser.add_argument('-m', '--WORD_EMBEDDING', type = str)
parser.add_argument('-d', '--EMBEDDING_DIM', type = int, default = 300)
parser.add_argument('-f', '--MININUM_DF', type = int, default = 1)
parser.add_argument('-v', '--DYNAMIC_VAR', type = int, default = 0)
args = parser.parse_args()
pos_data = args.POSITIVE_DATA
neg_data = args.NEGATIVE_DATA
word_embed = args.WORD_EMBEDDING
embed_dim = args.EMBEDDING_DIM
min_df = args.MININUM_DF
dyn_var = bool(int(args.DYNAMIC_VAR))

def build_data_cv(data_folder, cv=10, clean_string=True):
    """
    Loads data and split into 10 folds.
    """
    revs = []
    pos_file = data_folder[0]
    neg_file = data_folder[1]
    vocab = defaultdict(float)
    with open(pos_file, "rb") as f:
        for line in f:       
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev)
                # orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum  = {"y":1, 
                      "text": orig_rev,                             
                      "num_words": len(orig_rev.split()),
                      "split": np.random.randint(0,cv)}
            revs.append(datum)
    with open(neg_file, "rb") as f:
        for line in f:       
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev)
                # orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum  = {"y":0, 
                      "text": orig_rev,                             
                      "num_words": len(orig_rev.split()),
                      "split": np.random.randint(0,cv)}
            revs.append(datum)
    return revs, vocab
    
def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k), dtype='float32')            
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

def load_bin_vec(fname, vocab):
    """
    Loads 650x1 word vecs from Google (Mikolov) word2vec
    """
    with open(fname) as thefile:  
        word_vecs = {line.split(',')[0]: np.array([float(d) for d in line.strip().split(',')[1:]])\
         for line in thefile.readlines() if not line.startswith(',')}
        word_vecs = {key: word_vecs[key] for key in word_vecs.keys() if key in vocab}
    # word_vecs = {}
    # with open(fname, "rb") as f:
    #     header = f.readline()
    #     vocab_size, layer1_size = map(int, header.split())
    #     binary_len = np.dtype('float32').itemsize * layer1_size
    #     for line in xrange(vocab_size):
    #         word = []
    #         while True:
    #             ch = f.read(1)
    #             if ch == ' ':
    #                 word = ''.join(word)
    #                 break
    #             if ch != '\n':
    #                 word.append(ch)   
    #         if word in vocab:
    #            word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
    #         else:
    #             f.read(binary_len)
    return word_vecs

# def add_unknown_words(word_vecs, vocab, min_df=5, k=650):
def add_unknown_words(word_vecs, vocab, min_df=1, k=300, dynamic_variance = False):    
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    i = 0
    # suzi: this is truly data-driven 
    if dynamic_variance:
        # Variance of unif(-a,a): (2*a)^2 / 12 ==> a = sqrt(3*v)
        a = np.sqrt(3*np.mean(map(np.var, word_vecs.values())))
        print("Unknown vector initialization on [-", a, ",", a, "] driven from average variance of pre-trained vectors")
    else:
        a = 0.25
    # Suzi: It is important to consider the proportion of pre-trained words in tet
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            # word_vecs[word] = np.random.uniform(-0.25,0.25,k)            
            word_vecs[word] = np.random.uniform(-a,a,k)  
            i +=1 
    print "Added " + str(i) + " words of minimum df " + str(min_df) + " to W!"

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    # string = re.sub(r"/[A-Z]+", "", string)     
    string = re.sub(r'[A-z]+', 'R', string)
    # string = re.sub(r'[一-龥]+', 'H', string)
    # string = re.sub(r'[0-9]+(,[0-9]+)*', '#', string)
    string = re.sub(r"<sp>", "空", string)     
    string = re.sub(r"[^가-힣ᄀ-ᇂㄱ-ㅣ空A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"[0-9\.]+", "#", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip() if TREC else string.strip().lower()

def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r'[A-z]+', 'R', string)
    string = re.sub(r'[一-龥]+', 'H', string)
    string = re.sub(r'[0-9]+(,[0-9]+)*', '#', string)
    string = re.sub(r"<sp>", "空", string)     
    string = re.sub(r"[^가-힣ᄀ-ᇂㄱ-ㅣ空A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"[0-9\.]+", "#", string)     
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().lower()

if __name__=="__main__":    
    # w2v_file = "matrices/matrix-raw-skip-650.txt"
    # data_folder = ['data/kosac-polarity/raw.pos', 'data/kosac-polarity/raw.neg'] 
    w2v_file = word_embed
    data_folder = [pos_data, neg_data]
    print "loading data...",
    revs, vocab = build_data_cv(data_folder, cv=10, clean_string=False)
    # revs, vocab = build_data_cv(data_folder, cv=10, clean_string=True)
    max_l = np.max(pd.DataFrame(revs)["num_words"])
    print "data loaded!"
    print "positive sentences:" + pos_data
    print "negative sentences:" + neg_data
    print "number of sentences: " + str(len(revs))
    print "vocab size: " + str(len(vocab))
    # print " ".join(key for key in vocab.keys())
    print "max sentence length: " + str(max_l)
    print "loading word2vec vectors...",
    if w2v_file is None:
        w2v = {key: np.random.uniform(-0.25,0.25,embed_dim) for key in vocab.keys()}
    else:
        w2v = load_bin_vec(w2v_file, vocab)
    print "word2vec loaded!"
    print "embedding dimension: " + str(embed_dim)
    print "num words already in word2vec: " + str(len(w2v))
    add_unknown_words(w2v, vocab, min_df = min_df, k = embed_dim, dynamic_variance = dyn_var)
    W, word_idx_map = get_W(w2v, embed_dim)
    rand_vecs = {}
    add_unknown_words(rand_vecs, vocab, min_df = min_df, k = embed_dim, dynamic_variance = False)
    W2, _ = get_W(rand_vecs, embed_dim)
    cPickle.dump([revs, W, W2, word_idx_map, vocab], open("mr.p", "wb"))
    print "dataset created!"
    
