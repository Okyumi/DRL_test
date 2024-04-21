# REINFORCE: Lunar Lander

This project demonstrates the implementation of the REINFORCE algorithm, a policy gradient method, to solve the Lunar Lander environment in OpenAI Gym.

## Requirements

- Python 3.x
- OpenAI Gym
- PyTorch
- Matplotlib
- NumPy

## Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   ```

2. Install the required dependencies:

   ```bash
   pip install gym torch matplotlib numpy
   ```

## Usage

1. Run the main script to start training the agent:

   ```bash
   python main_lunar_lander_reinforce.py
   ```

   The script will train the agent for a specified number of episodes (default: 3000) and save the learning curve plot in the `plots/` directory.

2. Modify the hyperparameters in `main_lunar_lander_reinforce.py` if desired:

   - `n_games`: Number of episodes to train the agent (default: 3000).
   - `gamma`: Discount factor for future rewards (default: 0.99).
   - `lr`: Learning rate for the policy network (default: 0.0005).

## Project Structure

- `main_lunar_lander_reinforce.py`: The main script that sets up the environment, creates the agent, and trains it.
- `REINFORCE.py`: Contains the implementation of the REINFORCE algorithm and the policy network.
- `plots/`: Directory to store the learning curve plots.

## Algorithm

The REINFORCE algorithm is a policy gradient method that directly optimizes the policy network based on the rewards obtained from the environment. The key steps of the algorithm are:

1. Initialize the policy network with random weights.
2. For each episode:
   - Reset the environment and obtain the initial observation.
   - While the episode is not done:
     - Choose an action based on the current policy.
     - Take the action and observe the next state, reward, and done flag.
     - Store the reward and log probability of the chosen action.
   - Calculate the discounted returns (G) for each timestep.
   - Calculate the loss as the negative log probability of actions multiplied by the returns.
   - Perform gradient ascent on the policy network to maximize the expected return.
   - Reset the action and reward memory for the next episode.

## Results

The learning curve plot will be saved in the `plots/` directory after training is completed. The plot shows the running average of the previous 100 scores over the course of training.

## References

- [REINFORCE: Monte Carlo Policy Gradient - Andrej Karpathy](http://karpathy.github.io/2016/05/31/rl/)
- [OpenAI Gym: Lunar Lander](https://gym.openai.com/envs/LunarLander-v2/)

## License

This project is licensed under the [MIT License](LICENSE).
