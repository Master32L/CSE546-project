import torch.nn as nn
import torch.nn.functional as F

#%%

class Net1(nn.Module):
    def __init__(self, vocab_size, feature_size):
        super(Net1, self).__init__()
        self.vocab_size = vocab_size
        self.feature_size = feature_size
        self.encoder = nn.Embedding(self.vocab_size, self.feature_size)
        self.gru = nn.GRU(self.feature_size, self.feature_size, num_layers=2, batch_first=True)
        self.out = nn.Sequential(nn.LeakyReLU(),
                                 nn.Linear(self.feature_size * 2, 1))

    def forward(self, x):
        x = self.encoder(x)
        _, hidden = self.gru(x, None)
        return self.out(hidden.view(1, -1))

    def loss(self, prediction, label):
        loss_val = F.mse_loss(prediction, label)
        return loss_val

    def clip_grad(self, clip=50.0):
        nn.utils.clip_grad_norm_(self.encoder.parameters(), clip)
        nn.utils.clip_grad_norm_(self.gru.parameters(), clip)
