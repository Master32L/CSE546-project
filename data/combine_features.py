import pandas as pd
import string
import pickle
import numpy as np

#%%

df = pd.read_csv('yelp_academic_dataset_review.csv', usecols=['user_id', 'business_id', 'stars', 'useful', 'funny', 'cool'])
#print(len(df))
#df.useful = pd.to_numeric(df.useful, errors='coerce')
#df.useful = df.useful.astype(np.int32)
#df = df.dropna()
#print(len(df))
#
#idx = []
#for i, text in enumerate(df.text):
#    p = True
#    for char in text:
#        if char not in string.printable:
#            p = False
#            break
#    if p:
#        idx.append(i)
#print(len(idx))
#
#df.iloc[idx].to_csv('printable_review_useful.csv', index=False)
