import gym
import matplotlib.pyplot as plt
import numpy as np
from REINFORCE import PolicyGradientAgent

def plot_learning_curve(scores, x, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
        plt.plot(x, running_avg)
        plt.title('Running average of previous 100 scores')
        plt.savefig(figure_file)
        plt.close()

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    n_games = 3000
    agent = PolicyGradientAgent(gamma = 0.99, lr = 0.0005, input_dims=[8], n_actions=4)

    # saving for the file
    fname = 'REINFORCE' + 'lunar_lander_lr' + str(agent.lr) + '_' \
            + str(n_games) + 'games'
    figure_file = 'plots/' + fname + '.png'

    scores = []
    for i in range(n_games):
        done = False
        observation, _ = env.reset()
        score = 0
        while not done: # within one episode
            action = agent.choose_action(observation)
            observation_, reward, terminated, truncated, info = env.step(action)
            score += reward
            agent.store_reward(reward)
            observation = observation_
            done = (terminated or truncated)
        # after one episode we want agent to learn
        agent.learn()
        scores.append(score)

        avg_score = np.mean(scores[-100:])
        print('episode ', i, 'score %.2f' % score,
              'average score %.2f' % avg_score)
    # after all the iterations, we are going to start the plot
    x = [i+1 for i in range(len(scores))]
    plot_learning_curve(scores, x, figure_file)