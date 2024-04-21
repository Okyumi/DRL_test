import numpy as np


class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions): # input_shape is the dimension of the obs, n_actions is the dimension of the action
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape)) # unpacking a list represent the dimension of the obs
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_) # as a mask for setting the critic values for the new states to zero

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done
        
        self.mem_cntr += 1
    
    def sample_buffer(self, batch_size):
        # so choosing batchs_size number of transition tuples from index 0 to index max_mem
        max_mem_index = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem_index, batch_size) # this gives a list of indecies, with shape of batch_size

        states = self.state_memory[batch] # a list of states
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones