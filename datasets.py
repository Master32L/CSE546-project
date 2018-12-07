from torch.utils.data import Dataset
import torch
import pandas as pd
import string
import pickle
import numpy as np


#%%


class Vocab_Char(object):
    def __init__(self):
        self.ind2char = {i: c for i, c in enumerate(string.printable)}
        self.char2ind = {c: i for i, c in enumerate(string.printable)}

    def str2arr(self, string):
        return [self.char2ind[char] for char in string]

    def arr2str(self, arr):
        return ''.join([self.ind2char[int(i)] for i in arr])

    def __len__(self):
        return len(self.ind2char)


class Vocab_Char_Pad(object):
    def __init__(self):
        self.ind2char = {i+1: c for i, c in enumerate(string.printable)}
        self.ind2char[0] = '<PAD>'
        self.char2ind = {c: i+1 for i, c in enumerate(string.printable)}
        self.char2ind['<PAD>'] = 0

    def str2arr(self, string):
        return [self.char2ind[char] for char in string]

    def arr2str(self, arr):
        return ''.join([self.ind2char[int(i)] for i in arr])

    def __len__(self):
        return len(self.ind2char)


class Vocab_Word_Pad(object):
    def __init__(self):
        with open(f'./data/my_voc.pkl', mode='rb') as f:
            words = pickle.load(f)
        self.size = len(words) + 2
        self.ind2word = {i+1: w for i, w in enumerate(words)}
        self.ind2word[0] = '<PAD>'
        self.ind2word[self.size-1] = '<OUT>'
        self.word2ind = {w: i+1 for i, w in enumerate(words)}
        self.word2ind['<PAD>'] = 0
        self.word2ind['<OUT>'] = self.size - 1

    def str2arr(self, string):
        words = text2words(string)
        arr = []
        for word in words:
            if word in self.word2ind:
                arr.append(self.word2ind[word])
            else:
                arr.append(self.size - 1)
        return arr

    def arr2str(self, arr):
        return ' '.join([self.ind2word[i] for i in arr])

    def __len__(self):
        return self.size


#%%


class Char(Dataset):
    def __init__(self, split):
        super(Char, self).__init__()
        self.data = pd.read_csv('./data/' + split + '.csv')
        self.voc = Vocab_Char()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        review = self.data.text[idx]
        useful = self.data.useful[idx]
        return torch.LongTensor(self.voc.str2arr(review)), torch.FloatTensor([useful])


class Char_Pad(Dataset):
    def __init__(self, split, subset=None):
        super(Char_Pad, self).__init__()
        self.data = pd.read_csv('./data/' + split + '.csv')
        if subset is not None:
            self.data = self.data[:round(subset*len(self.data))]
        self.voc = Vocab_Char_Pad()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        review = self.data.text[idx]
        useful = self.data.useful[idx]
        return torch.LongTensor(self.voc.str2arr(review)), torch.FloatTensor([useful])


class Word_Pad(Dataset):
    def __init__(self, split, subset=None):
        super(Word_Pad, self).__init__()
        self.data = pd.read_csv('./data/' + split + '.csv')
        if subset is not None:
            self.data = self.data[:round(subset*len(self.data))]
        self.voc = Vocab_Word_Pad()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        review = self.data.text[idx]
        useful = self.data.useful[idx]
        return torch.LongTensor(self.voc.str2arr(review)), torch.FloatTensor([useful])


#%%


def text2words(text):
    temp = []
    for char in text:
        if char in string.punctuation:
            temp.append(' ')
        else:
            temp.append(char)
    return ''.join(temp).split()


def collate_pad(batch):
    lengths = np.array([len(sample[0]) for sample in batch])
    order = np.flip(np.argsort(lengths)) # sort sequences by length
    data = torch.zeros(len(batch), lengths.max(), dtype=torch.int64)
    labels = []
    for i, idx in enumerate(order):
        seq, label = batch[idx]
        labels.append(label)
        for j in range(len(seq)):
            data[i, j] = seq[j]
    return data, list(lengths[order]), torch.cat(labels)


def load_glove(model):
    dim = model.encoder.weight.shape[1]
    weight_path = f'./data/glove_weight_{dim}.pkl'
    with open(weight_path, mode='rb') as f:
        weight = pickle.load(f)
    model.encoder.weight.data = weight
    model.encoder.weight.requires_grad = False
