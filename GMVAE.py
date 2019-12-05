import torch
from torch import nn, optim
from torch.nn import functional as F
from fcnet import *
import math

class GMVAE(nn.Module):
    def __init__(self, K, sigma, input_dim, x_dim, w_dim, hidden_dim, hidden_layers, device):

        super(GMVAE, self).__init__()

        self.K = K
        self.sigma = sigma
        self.input_dim = input_dim
        self.x_dim = x_dim
        self.w_dim = w_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.device = device

        # Q_xw
        self.fc_q1 = FCNet(input_dim = self.input_dim, output_dim = self.hidden_dim, hidden_dim = self.hidden_dim, hidden_layers = self.hidden_layers, act_out = None)
        self.fc_mean_x = nn.Linear(self.hidden_dim, self.x_dim)
        self.fc_var_x = nn.Linear(self.hidden_dim, self.x_dim)
        self.fc_mean_w = nn.Linear(self.hidden_dim, self.w_dim)
        self.fc_var_w = nn.Linear(self.hidden_dim, self.w_dim)
        self.fc_qz = nn.Linear(self.hidden_dim, self.K)

        # Qz_x
        self.softmax_qz = nn.Softmax(dim=1)

        # Px_wz
        self.fc_x_wz = FCNet(input_dim = self.w_dim, output_dim = self.hidden_dim, hidden_dim = self.hidden_dim, hidden_layers = 0, act_out = "tanh")
        self.fc_x_means = nn.ModuleList()
        self.fc_x_vars = nn.ModuleList()
        self.x_mean_list = list()
        self.x_var_list = list()
        for i in range(self.K):
            self.fc_x_means.append(nn.Linear(self.hidden_dim, self.x_dim))
            self.fc_x_vars.append(nn.Linear(self.hidden_dim, self.x_dim))

        # Py_x
        self.fc_pyx = FCNet(input_dim = self.x_dim, output_dim = self.input_dim*2, hidden_dim = self.hidden_dim, hidden_layers = self.hidden_layers, act_out = None)


    def Q_xw(self, y):

        h = self.fc_q1(y)
        mean_x = self.fc_mean_x(h)
        var_x = torch.exp(self.fc_var_x(h))
        mean_w = self.fc_mean_w(h)
        var_w = torch.exp(self.fc_var_w(h))
        #qz = self.softmax_qz(self.fc_qz(h))
        #qz = F.softmax(self.fc_qz(h), dim=1)

        return mean_x, var_x, mean_w, var_w

    def Px_wz(self, w):
        h2 = self.fc_x_wz(w)
        self.x_mean_list = []
        self.x_var_list = []


        for i, l in enumerate(self.fc_x_means):
            self.x_mean_list.append(l(h2))
        for i, l in enumerate(self.fc_x_vars):
            a = l(h2)
            self.x_var_list.append(torch.exp(l(h2)))

        return self.x_mean_list, self.x_var_list


    def Py_x(self, x):
        params = self.fc_pyx(x)
        mean_y = params[:, 0:self.input_dim]
        var_y = torch.exp(params[:, self.input_dim:])

        return mean_y, var_y

    def reparameterize(self, mu, var, dim1, dim2):
        eps = torch.randn(dim1, dim2).to(self.device)
        return mu + eps*torch.sqrt(var)
    
    def forward(self, y):
        mean_x, var_x, mean_w, var_w = self.Q_xw(y)
        w_sample = self.reparameterize(mu = mean_w, var = var_w, dim1 = mean_w.size()[0], dim2 = mean_w.size()[1])
        x_sample = self.reparameterize(mu = mean_x, var = var_x, dim1 = mean_x.size()[0], dim2 = mean_x.size()[1])
        y_recons_mean, y_recons_var = self.Py_x(x_sample)
        x_mean_list, x_var_list = self.Px_wz(w_sample)
        qz = self.Qz_x(x_mean_list, x_var_list)
        y_recons = self.reparameterize(mu = y_recons_mean, var = y_recons_var, dim1 = y_recons_mean.size()[0], dim2 = y_recons_mean.size()[1])
        return y_recons, y_recons_mean, y_recons_var
    
    def Qz_x(self, x_mean_list, x_var_list):
        # KL  = 1/2(  logvar2 - logvar1 + (var1 + (m1-m2)^2)/var2  - 1 )
        x_mean_stack = torch.stack(x_mean_list)
        x_var_stack = torch.stack(x_var_list)
        K, bs, num_sample = x_mean_stack.size()
        qz = torch.zeros(bs, K, requires_grad=True, device = self.device)

        for i in range(num_sample):
            x_mean = x_mean_stack[:,:,i].view(bs, K)
            x_var = x_var_stack[:,:,i].view(bs, K)
            x_sample = self.reparameterize(mu = x_mean, var = x_var, dim1 = x_mean.size()[0], dim2 = x_mean.size()[1])
            qz = qz + x_sample/K
        #qz = qz/(torch.sum(x_sample, 1).view(bs,-1)*num_sample)
        qz = self.softmax_qz(qz/num_sample)

        return qz


        #return 0
