import torch.nn as nn

#%%

class Net1(nn.Module):
    def __init__(self, vocab_size, feature_size, num_layers):
        super(Net1, self).__init__()
        self.vocab_size = vocab_size
        self.feature_size = feature_size
        self.num_layers = num_layers
        self.encoder = nn.Embedding(vocab_size, feature_size)
        self.gru = nn.GRU(feature_size, feature_size,
                          num_layers=num_layers, batch_first=True)
        self.out = nn.Sequential(nn.LeakyReLU(),
                                 nn.Linear(feature_size*num_layers, 1))

    def forward(self, x):
        x = self.encoder(x)
        _, hidden = self.gru(x, None)
        return self.out(hidden.view(1, -1))

    def clip_grad(self, clip=50.0):
        nn.utils.clip_grad_norm_(self.encoder.parameters(), clip)
        nn.utils.clip_grad_norm_(self.gru.parameters(), clip)


class Net1_Pad(Net1):
    def __init__(self, vocab_size, feature_size, num_layers):
        super(Net1_Pad, self).__init__(vocab_size, feature_size, num_layers)

    def forward(self, x, lengths):
        x = self.encoder(x)
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
        _, hidden = self.gru(x, None)
        hidden = hidden.permute(1, 0, 2).contiguous()
        return self.out(hidden.view(-1, self.feature_size*self.num_layers))

