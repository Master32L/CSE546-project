import pandas as pd
import multiprocessing as mp
import pickle
from collections import Counter

import datasets


#%%

dim = 25
debug = False
batch_size = 1000

#%%

voc = []
with open(f'./glove.twitter.27B/glove.twitter.27B.{dim}d.txt', mode='rb') as f:
    for line in f:
        voc.append(line.decode().split()[0])

#%%

train = pd.read_csv('./data/train.csv')
train = ' '.join(list(train.text))
test = pd.read_csv('./data/test.csv')
test = ' '.join(list(test.text))
data = train + ' ' + test
data = data.lower()
if debug:
    data = data[:100000]

data_counter = Counter(datasets.text2words(data))
print(len(data_counter))

#        if word[-1] in string.punctuation:
#            if word[0] in string.punctuation:
#                temp.append(word[0])
#                temp.append(word[1:-1])
#                temp.append(word[-1])
#            else:
#                temp.append(word[:-1])
#                temp.append(word[-1])
#        elif word[0] in string.punctuation:
#            temp.append(word[0])
#            temp.append(word[1:])
#        else:
#            temp.append(word)

#%%

def find_notin(qin, qout, voc, counter):
    notin = []
    while True:
        words = qin.get()
        if words == 'STOP':
            break
        for word in words:
            if word not in voc:
                notin.append(word)
        with counter.get_lock():
            counter.value += 1
            print(counter.value)
    qout.put(notin)

#%%

print('Start processing...')

qin = mp.Queue()
qout = mp.Queue()
counter = mp.Value('i', 0)
ps = []
for i in range(4):
    p = mp.Process(target=find_notin, args=(qin, qout, voc, counter))
    p.start()
    ps.append(p)

batch = []
for word in data_counter.keys():
    batch.append(word)
    if len(batch) == batch_size:
        qin.put(batch)
        batch = []
if len(data_counter) % batch_size > 0:
    qin.put(batch) # last batch
for i in range(4):
    qin.put('STOP')

notin = []
for i in range(4):
    notin.extend(qout.get())

with open(f'notin_{dim}.pkl', mode='wb') as f:
    pickle.dump(notin, f)
print('Saved')

#%%

with open(f'notin_25.pkl', mode='rb') as f:
    notin = pickle.load(f)
print(len(notin))
print(notin[:20])

my_voc = [word for word in data_counter.keys() if word not in notin]
with open(f'my_voc.pkl', mode='wb') as f:
    pickle.dump(my_voc, f)
