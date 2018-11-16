import pandas as pd
import numpy as np

#%%

#df = pd.read_csv('printable_review_useful.csv')
df = pd.read_csv('filtered.csv')
n = len(df)
idx = np.arange(n)
np.random.shuffle(idx)
split = round(0.8 * n)

#%%

df.iloc[idx[:split]].to_csv('train.csv', index=False)
df.iloc[idx[split:]].to_csv('test.csv', index=False)
