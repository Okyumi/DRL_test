import gym

env = gym.make('LunarLander-v2')
obs = env.reset()
action = env.action_space.sample()
result = env.step(action)
print(result)
