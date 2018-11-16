from torch.utils.data import Dataset
import torch
import pandas as pd
import string


#%%


class Vocabulary(object):
    def __init__(self):
        self.ind2char = {i: c for i, c in enumerate(string.printable)}
        self.char2ind = {c: i for i, c in enumerate(string.printable)}

    def str2arr(self, string):
        return [self.char2ind[char] for char in string]

    def arr2str(self, arr):
        return ''.join([self.ind2char[int(i)] for i in arr])

    def __len__(self):
        return len(self.ind2char)


#%%


class Char_Filtered_Useful(Dataset):
    def __init__(self, split):
        super(Char_Filtered_Useful, self).__init__()
        self.data = pd.read_csv('data\\' + split + '.csv')
        self.voc = Vocabulary()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        review = self.data.text[idx]
        useful = self.data.useful[idx]
        return torch.LongTensor(self.voc.str2arr(review)), torch.FloatTensor([useful])
