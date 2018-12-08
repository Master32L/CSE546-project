import pickle
import torch
import numpy as np
import sys


#%%

dim = int(sys.argv[1])

#%%

with open(f'my_voc.pkl', mode='rb') as f:
    words = pickle.load(f)
word2ind = {w: i+1 for i, w in enumerate(words)}

#%%

weight = torch.randn(len(words)+2, dim)

#%%

with open(f'./glove.twitter.27B/glove.twitter.27B.{dim}d.txt', mode='rb') as f:
    count = 0
    for line in f:
        line = line.decode().split()
        if line[0] in words:
            weight[word2ind[line[0]],:] = torch.FloatTensor(np.array(line[1:], dtype=np.float32))
            count += 1
    print(count)
    print(len(words))


#%%

with open(f'glove_weight_{dim}.pkl', mode='wb') as f:
    pickle.dump(weight, f)

#%%

#with open(f'./glove.twitter.27B/glove.twitter.27B.200d.txt', mode='rb') as f:
#    count = 0
#    for line in f:
#        count += 1
#    print(count)
