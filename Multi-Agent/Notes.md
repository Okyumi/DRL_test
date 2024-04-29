25000 episodes, evaluate performance for 1000 itr

64 neurons for the hidden layer, with ReLu activation function

Structures
- Neural Networks -> DDPG w/o fancy initialization / noise
- Memory -> Local & global observations
- Agent -> Learn, choose action, update parameters, save/load models
- MADDPG -> store all the agents and call agent functions
- Run file -> train and evaluate without action noise

Neural Networks
- 64 neurons in each 2 layers
- Tanh for output of actor
- Linear output for critic
- Hiddent layers get relu activation
- Adam optimizer, e-4 for actor and e-3 for critic

Replay Buffer
- Combination of numpy arrays and lists of numpy arrays
- Arrays for combined observations of all the agents
- Reward and terminal memory also numpy arrays
- Store individual transitions(global obs and agent obs, etc.)
- Uniform sampling of buffer -> can use lists of arrays here, too
- Ready -> make sure we have at least 1 batch of memoris to sample 

Agent Class
- Init -> save relevant hyperparameters
- initiate neural nn
- choose action -> add noise to action & clamp & take eval flag
- Update network parameters -> slow moving copy of online network
- Save and load models

Critic: use target actor network to get new actions
 - evaluate those actions w/ target critic
Actor: sampled actions but get new actions for agent being updated
 - put those actions into critic and take negative mean for loss
Global critic means use global observations; each agent has a critic 

