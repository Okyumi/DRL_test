import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class PolicyNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(*input_dims, 128)# unpack a list for input
        self.fc2 = nn.Linear(128,128)
        self.fc3 = nn.Linear(128, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class PolicyGradientAgent():
    def __init__(self, lr, input_dims, gamma=0.99, n_actions=4):
        self.gamma = gamma
        self.lr = lr
        self.reward_memory = []
        self.action_memory = []

        self.policy = PolicyNetwork(self.lr, input_dims, n_actions)

    def choose_action(self, observation):
        # Assuming 'observation' is a list of numpy.ndarrays
        observation_np = np.array(observation)
        state = T.tensor([observation_np]).to(self.policy.device)
        #state = T.tensor([observation]).to(self.policy.device)
        probabilities = F.softmax(self.policy.forward(state), dim=1)
        action_probs = T.distributions.Categorical(probabilities) # telling numpy to model a random choice based on a custom defined distribution
        action = action_probs.sample()
        log_probs = action_probs.log_prob(action)
        self.action_memory.append(log_probs)

        return action.item() # open ai gym doesn't take pytorch tensor as an input
    
    def store_reward(self, reward):
        self.reward_memory.append(reward)
    
    def learn(self):
        self.policy.optimizer.zero_grad()

        # G_t = R_t+1 + gamma * R_t+2 + gamma**2 * R_t+3
        # G_t = sum from k=0 to k=T {gamma**k * R_t+k+1}
        G = np.zeros_like(self.reward_memory, dtype=np.float64)
        # if not using dynamic programming here
        for t in range(len(self.reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(t, len(self.reward_memory)):
                G_sum += self.reward_memory[k] * discount
                discount *= self.gamma
            G[t] = G_sum
        G = T.tensor(G, dtype=T.float).to(self.policy.device)
        # calculate the loss
        loss = 0
        for g, logprob in zip(G, self.action_memory):
            loss += -g * logprob
        # sum all of the loss and then backpropagate
        loss.backward()
        self.policy.optimizer.step()
        # after backpropagation we need to clean our action and reward memory, so
        self.action_memory = []
        self.reward_memory = []



        
