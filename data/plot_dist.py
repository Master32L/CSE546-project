import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#%%

df = pd.read_csv('printable_review_useful.csv')
df = df.drop([3371686])

##%%
#
#x = df.useful.unique()
#x.sort()
#x = x[1:]
#
##%%
#
#counts = []
#for i in x:
#    mask = df.useful.values == i
#    counts.append(mask.sum())
#plt.plot(x, counts)

#%%

mask = df.useful.values <= 20
mask.sum()

#%%

plt.hist(df.useful.values[mask])
plt.ylabel('number of samples')
plt.xlabel('number of useful votes')
plt.show()
