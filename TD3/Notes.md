Things to keep in Mind
- Try to solve the problem of 'Approximation error and overestimation bias'
- 2 critics (either shared input layer or be totally independent)
- will have to modify the learn function to take minimum of double q
- will have to have a way to delay policy updates

problem
- value based estimates are noisy, noise amplified by use of deep nn
- regular q-learning inherent problem, maximization of a noisy estimate -> overestimation
- overvalueing bad states -> suboptimal
- double DQN solution won't work due to policy similarities
2 problems: overestimation bias, and a high variance build-up

- clipped double q value (different from doubleQN)
- delayed policy updates
- adding clipped noise to output of target actor



