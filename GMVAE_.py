import torch
import torch.nn as nn

class GMVAE(nn.Module):
    def __init__(self, n_mel_channels):
        super(GMVAE, self).__init__()
        self.conv_layers = nn.Sequential(nn.Conv1d(n_mel_channels, 512, kernel_size=3), nn.ReLU(),
                                         nn.Conv1d(512, 512, 3), nn.ReLU())
        
        self.lstm = nn.LSTM(512, hidden_size = 256, num_layers = 2, bidirectional = True)
        self.linear_layer1 = nn.Linear(512, 16)
        self.linear_layer2 = nn.Linear(512, 16)
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.permute(0,2,1)
        x, _ = self.lstm(x)
        x = torch.mean(x, 1)
        mu = self.linear_layer1(x)
        log_var = self.linear_layer2(x)
        std = torch.exp(log_var)
        m = torch.distributions.normal.Normal(mu, std)
        
        return m.rsample((mu.size()[0], 16)) , log_var, mu
