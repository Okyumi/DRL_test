off-policy model free learning problem
- because for E[Q(st+1, at+1)] the at+1 = mu(s_t+1), deterministic nature, the expectation depends only on the environment, which is state transition probabilities; ranther than behavior policy beta; so we can using transitions generated from a different stochastic behavior policy beta
- if we assume deterministic policy, then the bellman equation is just defined in terms of the environment's dynamics, and we only have to sample those dynamics by way of a stochastic policy beta
- the Q value of the policy doesn't depend on the behavior policy (but for non-deterministic case, it depends)
- better to get a batch of uncorrelated transitions (most optimization algorithms assume that the samples are independently and identically distributed)

why target network?
- can lead to divergence by chasing a moving target
- to reduce the divergence 
- tao, sfoten the changing -> better learning stability

batch normalization
- input vector can have components that are expressed in different units -> widely different scales -> not conducive to training a deep nn
- manually scale the features using batch normalization so that inputs have a mean and viaraince of 1 (used in both actor and critic network)

Explore vs Exploit
- very important because the agent is updating a deterministic policy
- add some noise, ornstein-unlenbeck noise (model noise of  Brownian particle)

Key points:
- Actor and Critic 2 networks
- loss functions are related to each other. the update of the actor is going to be proportional to the gradinet of the critic as well as the actor by the chain rule

Code Structure
- class for actor network
- class for critic network
- class for action noise
- class for replay buffer
- class for agent functionality
    - memory, actor/critic networks, target nets, tau

Replay bUFFER
- Fixed size, overwrite early memories
 - sample memories uniformly
 - __init__()
 - store_transition()
 - no casting to pytorch tensors
 - using numpy arrays