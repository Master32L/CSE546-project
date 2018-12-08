from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn.functional as F
import pickle
import argparse
import time
import numpy as np

import datasets
import models


#%%


def trial(train_loader, test_loader, device, feature_dim, weight_decay, args):
    model = models.Net1_Pad(len(train_loader.dataset.voc),
                            feature_dim, 2).to(device)
    lr = 5e-4
    multiplier = 0.5
    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)

    train_losses = []
    test_losses = []
#    train_interval = len(train_loader) // 4 # print interval
    epoch = 0
    stuck = 0
    adjust = 0
    min_loss = 100
    while True:
        epoch += 1
        t1 = time.time()
        # train
        train_loss = 0
        for idx, (sample, lengths, target) in enumerate(train_loader):
            sample = sample.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            outputs = model(sample, lengths).squeeze()
            loss = F.mse_loss(outputs, target, reduction='sum')
            (loss / target.shape[0]).backward()
            model.clip_grad()
            optimizer.step()
            train_loss += float(loss)
#            if idx > 0 and idx % train_interval == 0:
#                print(f'{100*(idx+1)/len(train_loader):.2f}%  ',
#                      train_loss/(idx+1)/args.batch_size)
        train_loss /= len(train_loader.dataset)
        t2 = time.time()
        # test
        test_loss = 0
        with torch.no_grad():
            for idx, (sample, lengths, target) in enumerate(test_loader):
                sample = sample.to(device)
                target = target.to(device)
                outputs = model(sample, lengths).squeeze()
                test_loss += float(F.mse_loss(outputs, target,
                                              reduction='sum'))
        test_loss /= len(test_loader.dataset)
        t3 = time.time()
        # log
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        print('Epoch', epoch, 'train mse', train_loss, 'test mse', test_loss)
        # time
        if args.time:
            print(t2 - t1)
            print(t3 - t2)
            break
        # adjust lr
        if test_loss < min_loss:
            min_loss = test_loss
            stuck = 0
        else:
            stuck += 1
        if stuck == 4:
            adjust += 1
            stuck = 0
            lr *= multiplier
            optimizer = optim.Adam(model.parameters(), lr=lr,
                        weight_decay=weight_decay)
        if adjust == 3:
            break
    return train_losses, test_losses


#%%

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-v', '--version')
    group.add_argument('-t', '--time', action='store_true')
    parser.add_argument('-b', '--batch_size', type=int, default=128)

    args = parser.parse_args()

#%%

    device = torch.device('cuda')
#    device = torch.device('cpu')

    train_data = datasets.Word_Pad('train')
    test_data = datasets.Word_Pad('test')
    train_loader = DataLoader(train_data, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True,
                              collate_fn=datasets.collate_pad)
    test_loader = DataLoader(test_data, batch_size=args.batch_size,
                             shuffle=True, num_workers=2, pin_memory=True,
                             collate_fn=datasets.collate_pad)

    all_train_loss, all_test_loss = [], []
    all_feature_dim, all_weight_decay = [], []
    count = 0
    try:
        for d in [256]:
            count += 1
            if args.time:
                d = 256
            print('\nTrial', count, 'dim', d,
                  'decay', 1e-4)
            train_losses, test_losses = trial(train_loader, test_loader,
                                              device, d, 1e-4, args)
            all_train_loss.append(train_losses)
            all_test_loss.append(test_losses)
            all_feature_dim.append(d)
            all_weight_decay.append(1e-4)
            if args.time:
                break
    finally:
        with open(f'./result/search-{args.version}.pkl', mode='wb') as f:
            pickle.dump([all_train_loss, all_test_loss,
                         all_feature_dim, all_weight_decay], f)