from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import pickle
import os
import numpy as np
import time

import datasets
import models


#%%


def my_collate(batch):
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


#%%

if __name__ == '__main__':

    epochs = 2

#%%

    device = torch.device('cuda')
#    device = torch.device('cpu')

    train_data = datasets.Char_Filtered_Useful_Pad('train', 0.1)
    test_data = datasets.Char_Filtered_Useful_Pad('test')
    train_loader = DataLoader(train_data, batch_size=256, shuffle=True,
                              num_workers=4, pin_memory=True,
                              collate_fn=my_collate)
    test_loader = DataLoader(test_data, batch_size=4, shuffle=True,
                             num_workers=4, pin_memory=True,
                             collate_fn=my_collate)

    model = models.Net1_Pad(len(train_data.voc), 64, 1).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

#%%

#    for idx, (sample, lengths, target) in enumerate(train_loader):
##        print(sample.shape, sample.dtype)
##        print(target.shape, target.dtype)
##        model(sample, lengths)
#        outputs = model(sample, lengths).squeeze()
#        print(outputs.shape)
#        loss = model.loss(outputs, target)
#        print(loss)
#        loss.backward()
#        model.clip_grad()
#        optimizer.step()
#        break

#%%

    try:
        train_losses = []
        test_losses = []
        for epoch in range(epochs):
            print('\nEpoch:', epoch + 1)

            train_loss = 0
            start = time.time()
            for idx, (sample, lengths, target) in enumerate(train_loader):
                sample = sample.to(device)
                target = target.to(device)
                optimizer.zero_grad()
                outputs = model(sample, lengths).squeeze()
                loss = model.loss(outputs, target)
                loss.backward()
                model.clip_grad()
                optimizer.step()
                train_loss += float(loss)
            train_losses.append(train_loss / len(train_data))
            print('Training loss:', train_losses[-1])
            print(time.time() - start)
            break

#            test_loss = 0
#            start = time.time()
#            with torch.no_grad():
#                for idx, (sample, lengths, target) in enumerate(test_loader):
#                    sample = sample.to(device)
#                    target = target.to(device)
#                    outputs = model(sample, lengths).squeeze()
#                    test_loss += float(model.loss(outputs, target))
#            test_losses.append(test_loss / len(test_data))
#            print('Test loss:', test_losses[-1])
#            print(time.time() - start)
#            break

    finally:
        pass
