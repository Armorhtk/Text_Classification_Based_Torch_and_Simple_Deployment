import os
import re
import jieba
import torch
import random
import numpy as np
from torchtext.legacy import data
from torchtext.vocab import GloVe,Vectors
from Config import *

torch.manual_seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True

def x_tokenize(x):
    str = re.sub('[^\u4e00-\u9fa5]', "", x)
    return jieba.lcut(str)

TEXT = data.Field(sequential=True,
                  tokenize=x_tokenize,
                  fix_length=max_length,
                  use_vocab=True)

LABEL = data.Field(sequential=False,
                   use_vocab=False)

train, dev, test = data.TabularDataset.splits(path=os.path.join(data_path,task_name),
                                              train=train_file,
                                              validation=valid_file,
                                              test=test_file,
                                              format='csv',
                                              skip_header=True,
                                              csv_reader_params={'delimiter':','},
                                              fields=[("text",TEXT),('label',LABEL)])

# 以下两种指定预训练词向量的方式等效
if load_embedding == "None":
    TEXT.build_vocab(train)
elif load_embedding == "w2v":
    cache = '.vector_cache'
    if not os.path.exists(cache):
        os.mkdir(cache)
    vectors = Vectors(name=w2v_file,cache=cache)
    TEXT.build_vocab(train, vectors=vectors)
elif load_embedding == "glove":
    cache = '.vector_cache'
    if not os.path.exists(cache):
        os.mkdir(cache)
    vectors = Vectors(name=glove_file, cache=cache)
    TEXT.build_vocab(train, vectors=vectors)
else:
    TEXT.build_vocab(train)

train_iter, val_iter, test_iter = data.BucketIterator.splits(datasets = (train,dev,test),
                                                             batch_size = batch_size,
                                                             shuffle=True,
                                                             sort=False,
                                                             sort_within_batch=False,
                                                             repeat=False)

def getTEXT():
    return TEXT

def getLabel():
    return LABEL

def getIter():
    return train_iter, val_iter, test_iter



if __name__=="__main__":
    # train_iter, val_iter, test_iter = getIter()
    # for batch in train_iter:
    #     print(batch.text.shape)
    # 会有一个加载过程。
    cache = '.vector_cache'
    vectors = Vectors(name=w2v_file, cache=cache)