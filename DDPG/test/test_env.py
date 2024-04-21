import gym

env = gym.make('LunarLanderContinuous-v2')
obs = env.reset()
print(obs)
action = env.action_space.sample()
result = env.step(action)
print(result)
print(env.observation_space.shape)