from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import pickle
import os
import numpy as np

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

    version = '1206-2'
    save_path = './result/' + version + '/'
    os.makedirs(save_path)
    epochs = 1000
    batch_size = 128
    lmd = 1e-3 # l1 reg

#%%

    device = torch.device('cuda')
#    device = torch.device('cpu')

    train_data = datasets.Char_Filtered_Useful_Pad('train')
    test_data = datasets.Char_Filtered_Useful_Pad('test')
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True,
                              collate_fn=my_collate)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True,
                             num_workers=2, pin_memory=True,
                             collate_fn=my_collate)

    model = models.Net1_Pad(len(train_data.voc), 64, 1).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)

#%%

    try:
        train_losses = []
        test_losses = []
        train_interval = len(train_loader) // 4 # print interval
        for epoch in range(epochs):
            print('\nEpoch:', epoch + 1)
            # train
            train_loss = 0
            for idx, (sample, lengths, target) in enumerate(train_loader):
                sample = sample.to(device)
                target = target.to(device)
                optimizer.zero_grad()
                outputs, hidden = model(sample, lengths)
                loss = model.loss(outputs.squeeze(), target)
                (loss + lmd*hidden.abs().sum()).backward()
                model.clip_grad()
                optimizer.step()
                train_loss += float(loss)
                if idx > 0 and idx % train_interval == 0:
                    print(f'{100*(idx+1)/len(train_loader):.2f}%  ',
                          train_loss/(idx+1)/batch_size)
            train_loss /= len(train_data)
            print('Training loss:', train_loss)
            torch.save(model.state_dict(), f'{save_path}{epoch+1}.pth')
            print('Model saved')
            # test
            test_loss = 0
            with torch.no_grad():
                for idx, (sample, lengths, target) in enumerate(test_loader):
                    sample = sample.to(device)
                    target = target.to(device)
                    outputs = model(sample, lengths)[0].squeeze()
                    test_loss += float(model.loss(outputs, target))
            test_loss /= len(test_data)
            print('Test loss:', test_loss)
            # log
            train_losses.append(train_loss)
            test_losses.append(test_loss)

    finally:
        with open(save_path + 'log.pkl', mode='wb') as f:
            pickle.dump({'training loss': train_losses,
                         'test loss': test_losses}, f)
        print('Log saved')
