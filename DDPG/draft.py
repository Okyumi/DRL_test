class CriticNetwork(nn.Module):
    def __init__(self, fc1_dims, fc2_dims, output_size, input_dims, n_actions,
                 ):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions


        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)

        # using layer normalization instead of batch normalization
        self.bn1 = nn.LayerNorm(self.fc1_dims)
        self.bn2 = nn.LayerNorm(self.fc2_dims)
        # we need a layer to handle the input of our action values
        self.action_value = nn.Linear(self.n_actions, self.fc2_dims) # the action is not included untill reach the second hidden layer
        # we need a layer to join everything together to get the actual critic values
        self.q = nn.Linear(self.fc2_dims, output_size)

        # do the initialiation of the weights and biases
        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1,f1)
        
        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2,f2)
        self.fc2.bias.data.uniform_(-f2,f2)

        f3 = 0.003
        self.q.weight.data.uniform_(-f3, f3)
        self.q.bias.data.uniform_(-f3, f3)

        f4 = 1./np.sqrt(self.action_value.weight.data.size()[0])
        self.action_value.weight.data.uniform_(-f4, f4)
        self.action_value.bias.data.uniform_(-f4,f4)

        # set the optimizer, deal with the device
        self.optimizer = optim.Adam(self.parameters(), lr=beta, 
                                    weight_decay=0.01)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, state, action):
        state_value = self.fc1(state)
        #batch normalization
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)
        action_value = self.action_value(action)
        state_action_value = F.relu(T.add(state_value, action_value)) # activate them together
        state_action_value = self.q(state_action_value)

        return state_action_value