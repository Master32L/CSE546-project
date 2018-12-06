

#%%

with open('./glove.twitter.27B/glove.twitter.27B.25d.txt', mode='rb') as f:
    for i, line in enumerate(f):
        print(str(line)[:10])
        if i == 20: break
