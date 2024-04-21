import torch as T
import torch.nn.functional as F
from networks import ActorNetwork, CriticNetwork
from noise import OUActionNoise
from buffer import ReplayBuffer
import numpy as np

class Agent():
    def __init__(self, alpha, beta, input_dims, tau, n_actions, gamma=0.99,
                 max_size=1000000, fc1_dims=400, fc2_dims=300,
                 batch_size=64):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta

        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.noise = OUActionNoise(mu=np.zeros(n_actions))

        self.actor = ActorNetwork(alpha, input_dims, fc1_dims, fc2_dims, 
                                  n_actions=n_actions, name='actor')
        self.critic = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims,
                                    n_actions=n_actions, name='critic')
        self.target_actor = ActorNetwork(alpha, input_dims, fc1_dims, fc2_dims,
                                         n_actions=n_actions, name='target_actor')
        self.target_critic = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims,
                                           n_actions=n_actions, name='target_critic')
        
        self.update_network_parameters(tau=1)
    
    def choose_action(self, observation):
        # dont want to handle the statistic through forward pass, since we are using layernorm here, so eval mode
        self.actor.eval()
        observation_np = np.array(observation)
        state = T.tensor([observation_np], dtype=T.float).to(self.actor.device)
        # forward pass
        mu = self.actor.forward(state).to(self.actor.device)
        # add some noise here
        mu_prime = mu + T.tensor(self.noise(), dtype=T.float).to(self.actor.device)
        # set it back to the training mode
        self.actor.train()
        # return numpy array rather than tensor
        return mu_prime.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state,action,reward,state_,done)
    
    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()
    
    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()
    
    def learn(self):
        # deal with the situation where we have not fill the memory yet to the batch size
        if self.memory.mem_cntr < self.batch_size:
            return
        
        states, actions, rewards, states_, dones = \
                        self.memory.sample_buffer(self.batch_size)
        # saved as numpy array -> pytorch tensor
        states = T.tensor(states, dtype=T.float).to(self.actor.device)
        states_ = T.tensor(states_, dtype=T.float).to(self.actor.device)
        actions = T.tensor(actions, dtype=T.float).to(self.actor.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.actor.device)
        dones = T.tensor(dones).to(self.actor.device)

        # for state_, using the target action network to get the actions',
        target_actions = self.target_actor.forward(states_)
        critic_value_ = self.target_critic.forward(states_, target_actions)
        # current critic value
        critic_value = self.critic.forward(states, actions)

        critic_value_[dones] = 0.0 # collapse the dimension
        critic_value_ = critic_value_.view(-1)

        target = rewards + self.gamma*critic_value_
        target = target.view(self.batch_size, 1) # add a dimension to it
        # calculate the loss for the critic first
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()
        # calculate the loss for the actor now
        self.actor.optimizer.zero_grad()
        actor_loss = -self.critic.forward(states, self.actor.forward(states))
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters() # so actually soft update the target network every time you learn
    
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        
        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        actor_state_dict = dict(actor_params)
        critic_state_dict = dict(critic_params)
        target_actor_dict = dict(target_actor_params)
        target_critic_dict = dict(target_critic_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone()+ \
                                    (1-tau)*target_critic_dict[name].clone()
        
        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                                    (1-tau)*target_actor_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)





