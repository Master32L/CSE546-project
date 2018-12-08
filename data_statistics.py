import pandas as pd
import numpy as np

#%%

test = pd.read_csv('./data/test.csv')
train = pd.read_csv('./data/train.csv')

#%%

test.useful.values.var()
train.useful.values.var()
#np.concatenate([test.useful.values, train.useful.values]).var()
np.arange(1, 21).var()

#%%

m = train.useful.values.mean()
np.square(test.useful.values - m).mean()
