from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn.functional as F
import pickle
import os
import argparse
import time

import datasets
import models


#%%

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-v', '--version')
    group.add_argument('-t', '--time', action='store_true')
    parser.add_argument('dim', type=int)
    parser.add_argument('-f', '--fix', action='store_true')
    parser.add_argument('-b', '--batch_size', type=int, default=512)
    parser.add_argument('-n', '--num_worker', type=int, default=2)
    parser.add_argument('-l', '--lr', type=float, default=1e-3)
    parser.add_argument('-d', '--decay', type=float, default=1e-4)

    args = parser.parse_args()

    if not args.time:
        save_path = './result/' + args.version + '/'
        os.makedirs(save_path)

#%%

    device = torch.device('cuda')
#    device = torch.device('cpu')

    train_data = datasets.Word_Pad('train')
    test_data = datasets.Word_Pad('test')
    train_loader = DataLoader(train_data, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_worker,
                              pin_memory=True,
                              collate_fn=datasets.collate_pad)
    test_loader = DataLoader(test_data, batch_size=args.batch_size,
                             shuffle=True, num_workers=args.num_worker,
                             pin_memory=True,
                             collate_fn=datasets.collate_pad)

    model = models.Net1_Pad(len(train_data.voc), args.dim, 1)
    datasets.load_glove(model, args.fix)
    model.to(device)

    lr = args.lr
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.decay)

#%%

    train_losses = []
    test_losses = []
    try:
        train_interval = len(train_loader) // 4 # print interval
        epoch = 0
        stuck = 0
        adjust = 0
        min_loss = 100
        while True:
            epoch += 1
            print('\nEpoch:', epoch)
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
                if idx > 0 and idx % train_interval == 0:
                    print(f'{100*(idx+1)/len(train_loader):.2f}%  ',
                          train_loss/(idx+1)/args.batch_size)
            train_loss /= len(train_data)
            print('Training loss:', train_loss)
            if not args.time:
                torch.save(model.state_dict(), f'{save_path}{epoch}.pth')
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
            test_loss /= len(test_data)
            print('Test loss:', test_loss)
            t3 = time.time()
            # log
            train_losses.append(train_loss)
            test_losses.append(test_loss)
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
            if stuck > 3:
                adjust += 1
                lr *= 0.1
                optimizer = optim.Adam(model.parameters(), lr=lr,
                            weight_decay=1e-4)
                print(f'lr changed to {lr}')
            if adjust == 3:
                break

    finally:
        if not args.time:
            with open(save_path + 'log.pkl', mode='wb') as f:
                pickle.dump({'training loss': train_losses,
                             'test loss': test_losses}, f)
            print('Log saved')
