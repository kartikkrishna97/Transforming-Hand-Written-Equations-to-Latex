import pandas as pd 
import numpy as np 
import nltk

path1 = 'col_774_A4_2023/HandwrittenData/train_hw.csv'

def preprocess_text(path):
    x = pd.read_csv(path)
    x = x['formula']
    X_train_new = []
    for i in range(len(x)):
      X_train_new.append(["sos"]+x[i].split()+["eos"])
    
    word2index = {"sos":0,"eos":1,"<pad>":2} 
    n = 3
    for lst in X_train_new:
      for i in range(len(lst)):
        if lst[i] not in word2index:
          word2index[lst[i]]=n
          n+=1
    index2word = {v: k for k, v in word2index.items()}

    for lst in X_train_new:
      for i in range(len(lst)):
        lst[i] = word2index[lst[i]]

          
    return X_train_new, word2index, index2word

X_train, word2index, index2word = preprocess_text(path1)






