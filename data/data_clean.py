import pandas as pd
import string
import pickle
import numpy as np

#%%

#df = pd.read_csv('yelp_academic_dataset_review.csv', usecols=['text'])
#
##%%
#
#good = []
#nontext = []
#try:
#    for i, text in enumerate(df.text[4744944:]):
#        if isinstance(text, str):
#            g = True
#            for char in text:
#                if char not in string.printable:
#                    g = False
#                    break
#            if g:
#                good.append(i)
#        else:
#            nontext.append(i)
#except:
#    print(i, text)
#
##%%
#
#with open('good.pkl', mode='wb') as f:
#    pickle.dump(good, f)
#
##%%
#
#df = pd.read_csv('printable_review.csv')

#%%
#
#df = pd.read_csv('yelp_academic_dataset_review.csv')
#int_cols = ['funny', 'stars', 'useful', 'cool']
#for col in df.columns:
#    if col in int_cols:
#        print(col)
#        df[col] = pd.to_numeric(df[col], errors='coerce')
#        df[col] = df[col].astype(np.int32)
#    else:
#        df[col] = df[col].astype('object')
#df = df.dropna()

#%%

df = pd.read_csv('yelp_academic_dataset_review.csv', usecols=['useful', 'text'])
print(len(df))
df.useful = pd.to_numeric(df.useful, errors='coerce')
df.useful = df.useful.astype(np.int32)
df = df.dropna()
print(len(df))

idx = []
for i, text in enumerate(df.text):
    p = True
    for char in text:
        if char not in string.printable:
            p = False
            break
    if p:
        idx.append(i)
print(len(idx))

df.iloc[idx].to_csv('printable_review_useful.csv', index=False)

#%%

#df = pd.read_csv('fix.csv', dtype={'useful': np.int32})
#
##%%
#
#def check(df):
#    for text in df.text:
#        for char in text:
#            if char not in string.printable:
#                print('wrong')
#                return
#check(df)
