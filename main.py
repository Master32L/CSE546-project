from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import pickle
import os

import datasets
import models


if __name__ == '__main__':

#%%

    version = 'debug'
    save_path = 'result\\' + version + '\\'
    os.makedirs(save_path)
    epochs = 1000

#%%

    device = torch.device('cuda')
    #device = torch.device('cpu')

    train_data = datasets.Char_Filtered_Useful('train')
    test_data = datasets.Char_Filtered_Useful('test')
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True,
                             num_workers=4, pin_memory=True)

    model = models.Net1(len(train_data.voc), 128).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

#%%

    try:
        train_losses = []
        test_losses = []
        for epoch in range(epochs):
            print('\nEpoch:', epoch + 1)

            train_loss = 0
            for idx, (sample, target) in enumerate(train_loader):
                sample = sample.to(device)
                target = target.to(device)
                optimizer.zero_grad()
                loss = model.loss(model(sample), target)
                loss.backward()
                model.clip_grad()
                optimizer.step()
                train_loss += float(loss)
                if idx % 400 == 0:
                    print(f'{100*idx/len(train_data)}%  {train_loss/(idx+1)}')
            train_losses.append(train_loss / len(train_data))
            print('Training loss:', train_losses[-1])
            torch.save(model.state_dict(), f'{save_path}{epoch+1}.pth')
            print('Model saved')

            test_loss = 0
            with torch.no_grad():
                for idx, (sample, target) in enumerate(test_loader):
                    sample = sample.to(device)
                    target = target.to(device)
                    test_loss += float(model.loss(model(sample), target))
                    print(f'{100*idx/len(train_data)}%  {train_loss/(idx+1)}')
            test_losses.append(test_loss / len(test_data))
            print('Test loss:', test_losses[-1])

    finally:
        torch.save(model.state_dict(), f'{save_path}{epoch+1}.pth')
        print('\nLatest model saved')
        with open(save_path + 'log.pkl', mode='wb') as f:
            pickle.dump({'training_loss': train_losses,
                         'test_loss': test_losses}, f)
        print('Log saved')
