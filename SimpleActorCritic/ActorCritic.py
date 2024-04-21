import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ActorCriticNetwork(nn.Module):
    def __init__(self, lr, input_dim, n_actions, fc1_dims=256, fc2_dims=256):
        super(ActorCriticNetwork, self).__init__()
        self.fc1 = nn.Linear(*input_dim, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        # two different output
        self.pi = nn.Linear(fc2_dims, n_actions)
        self.v = nn.Linear(fc2_dims, 1) # value network
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        # send network to device
        self.to(self.device)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        pi = self.pi(x)
        v = self.v(x)

        return (pi, v)
    
#let's handle the agent
class Agent():
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions, gamma=0.99):
        self.gamma = gamma
        self.lr = lr # saved for filename
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        
        