# offpolicy actor critic
- maximize the exploration but still maximize the reward in long term
- agent isn't deterministic
- DDPG quite sensitive to hyperparameter tuning
- temperature to control the entropy term (higher temp->higher entropy->higher exploration)
- actor, critic, value network
- have to modify our cost function
- will model the policy as a probability distribution

## Implementation Notes
- include entropy by scaling reward
- use the replay buffer in the learn function


- If we alternate between the soft policy evaluation, the application of the bellman update equation, and the minimization of the KL divergence, then an arbitrary policy Pi can be made to converge into the optimal policy pi star such that the q value is universally better. 