import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#%%

df = pd.read_csv('printable_review_useful.csv')

#%%

#plt.hist(df.useful.values, bins=[0, 10, 20, 50, 100, 1000, 2000])

#%%

#lens = []
#for text in df.text:
#    lens.append(len(text))

#plt.hist(lens)

#arr = np.array(lens)

#mask = arr > 1000
#mask.sum()
#votes = df.useful.iloc[mask]

#%%

#counts = []
#x = np.arange(50) + 1
#for i in x:
#    mask = df.useful.values == i
#    counts.append(mask.sum())

#%%

#mins = []
#for i in (np.arange(20) + 1):
#    mask = df.useful.values == i
#    s = arr[mask]
#    s.sort()
#    mins.append(s[-2500])
#plt.plot(mins)

#%%

full_idx = np.arange(len(df))
out_idx = []
for i in (np.arange(20) + 1):
    mask = df.useful.values == i
    idx = full_idx[mask]
    np.random.shuffle(idx)
    out_idx.extend(list(idx[:2500]))

#%%

df.iloc[out_idx].to_csv('filtered.csv', index=False)
