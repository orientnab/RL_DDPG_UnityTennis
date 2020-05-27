# RL DDPG UnityTennis
Implementation of a Deep Deterministic Policy Gradient method for solving Unity Reacher Environment.

This project is part of [Udacity Deep Reinforcement Learning](https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet) Nanodegree.

## Unity Tennis Environment

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single score for each episode.

## Solving the Environment
The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

## Getting Started
You can download the environment from one of the links below.
You need only select the environment that matches your operating system.

Option for Linux, Mac and Windows can be found in [Udacity Deep Reinforcement Learning repository](https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet).

It is also possible to train the agent in AWS. Full instructions in the aforementioned repository.

## Dependencies
You will need in your system:
- Jupyter
- Pytorch
- Numpy
- Some extras such as Matplotlib

## Instructions
Follow the instructions in `Tennis.ipynb` to get started with training the agent.

## Learning Algorithm
The algorithm used is Deep Deterministic Policy Gradient (DDPG).
Most of the code is based on example for solving the [Reacher Environment](https://github.com/orientnab/RL_DDPG_UnityReacher), and a full description of the algorithm can be found there.
For the choice of hyperparameters we followed [ishgirwan](https://github.com/ishgirwan/udacity_drlnd/tree/master/Collaboration%20and%20Competition).

| Hyperparameter | Value |
|---|---:|
| Replay buffer size | 1e6 |
| Replay batch size | 256 |
| Actor hidden units | 384, 384 |
| Actor critic units | 384, 384 |
| Actor learning rate | 1e-3 |
| Critic learning rate | 1e-3 |
| Learning timestep interval | 1 |
| Learning passes | 10 |
| Target update mix | 2e-3 |
| Discount factor | 0.99 |
| Ornstein-Uhlenbeck, mu | 0 |
| Ornstein-Uhlenbeck, theta | 0.15 |
| Ornstein-Uhlenbeck, sigma | 0.1 |
| Noise decay per timestep| 1e-6 |
| Max episodes | 1000 |
| Max steps | 1000 |

## Important Notes
- The Ornstein-Uhlenbeck sigma value must be very low, otherwise the noise is too intense and the model struggles to converge.
- It is also important to keep the learning rates below 1e-3.
- Adding a third layer massively slows down convergence.

## Future Work and Improvements
- The problem is very sensitive to the choice of hyperparameters. It would be ideal to find a more systematic way to find their optimal values.
- A Prioritized Experience Replay could be implemented to further enhance the performance of the agent.
- We could implement a Multi Agent DDPG algorithm.
