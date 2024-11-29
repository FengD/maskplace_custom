import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config


grid = config.grid

class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 8, 1),
            nn.ReLU(),
            nn.Conv2d(8, 8, 1),
            nn.ReLU(),
            nn.Conv2d(8, 1, 1),
        )
    def forward(self, x):
        return self.cnn(x)


class MyCNNCoarse(nn.Module):
    def __init__(self, res_net, device):
        super(MyCNNCoarse, self).__init__()
        self.cnn = res_net.to(device)
        self.cnn.fc = torch.nn.Linear(512, 16*7*7)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding = 1), #14
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, 3, stride=2, padding=1, output_padding = 1), #28
            nn.ReLU(),
            nn.ConvTranspose2d(4, 2, 3, stride=2, padding=1, output_padding = 1), #56
            nn.ReLU(),
            nn.ConvTranspose2d(2, 1, 3, stride=2, padding=1, output_padding = 1), #112
            nn.ReLU(),
            nn.ConvTranspose2d(1, 1, 3, stride=2, padding=1, output_padding = 1), #224
        )
    def forward(self, x):
        x = self.cnn(x).reshape(-1, 16, 7, 7)
        return self.deconv(x)


class Actor(nn.Module):
    def __init__(self, cnn, gcn, cnn_coarse):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(config.num_emb_state, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, grid * grid)
        self.cnn = cnn
        self.cnn_coarse = cnn_coarse
        self.gcn = None
        self.softmax = nn.Softmax(dim=-1)
        self.merge = nn.Conv2d(2, 1, 1)

    def forward(self, x, graph = None, cnn_res = None, gcn_res = None, graph_node = None):
        if not cnn_res:
            cnn_input = x[:, 1+grid*grid*1: 1+grid*grid*5].reshape(-1, 4, grid, grid)
            mask = x[:, 1+grid*grid*2: 1+grid*grid*3].reshape(-1, grid, grid)
            mask = mask.flatten(start_dim=1, end_dim=2)
            cnn_res = self.cnn(cnn_input)
            coarse_input = torch.cat((x[:, 1: 1+grid*grid*2].reshape(-1, 2, grid, grid),
                                        x[:, 1+grid*grid*3: 1+grid*grid*4].reshape(-1, 1, grid, grid)
                                        ),dim= 1).reshape(-1, 3, grid, grid)
            cnn_coarse_res = self.cnn_coarse(coarse_input)
            cnn_res = self.merge(torch.cat((cnn_res, cnn_coarse_res), dim=1))
        net_img = x[:, 1+grid*grid: 1+grid*grid*2]
        net_img = net_img + x[:, 1+grid*grid*2: 1+grid*grid*3] * 10
        net_img_min = net_img.min() + config.soft_coefficient
        mask2 = net_img.le(net_img_min).logical_not().float()

        x = cnn_res
        x = x.reshape(-1, grid * grid)
        x = torch.where(mask + mask2 >=1.0, -1.0e10, x.double())
        x = self.softmax(x)

        return x, cnn_res


class Critic(nn.Module):
    def __init__(self, cnn, gcn, cnn_coarse, res_net):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(64, 64)  # 64, 512
        self.fc2 = nn.Linear(64, 64)  # 512, 64
        self.state_value = nn.Linear(64, 1)
        self.pos_emb = nn.Embedding(1400, 64)
        self.cnn = cnn
        self.gcn = gcn
    def forward(self, x, graph = None, cnn_res = None, gcn_res = None, graph_node = None):
        x1 = F.relu(self.fc1(self.pos_emb(x[:, 0].long())))
        x2 = F.relu(self.fc2(x1))
        value = self.state_value(x2)
        return value
    
