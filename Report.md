[//]: # (Image References)

[image1]: ./img/untrained.gif "Untrained Agent"
[image2]: ./img/trained.gif "Trained Agent"
[image3]: ./img/dqn.png "DQN Scores"
[image4]: ./img/dqn-scores.png "DQN Scores List"
[image5]: ./img/double-dqn-scores.png "Double DQN Scores List"
[image6]: ./img/duel-dqn-scores.png "Dueling DQN Scores List"
[image7]: ./img/duel-double-dqn-scores.png "Dueling Double DQN Scores List"

# Report

## Introduction
The project consists of three programming files and a pre-built Unity-ML Agents enviroment.

The goal of the project is to train an agent to collect as many yellow bananas as possible while avoiding blue bananas. This is achieved through reinforcement learning. The agent receives a reward of +1 for every yellow banana and a reward of -1 for every blue banana.
The agent may use any of 4 actions (move forward, move left, move right, move backward) at any given time step. While the action space is relatively small and discrete, the observed state space of the environment is rather large (`n=37`). Therefore, we cannot use a traditional Q-Table approach like Sarsa, Sarsamax, or expected Sarsa. Instead the agent is trained by a neural network with multiple hidden layers; this is called a Deep Q-Network [DQN]. 

The agent is trained successfully to earn an average cumulative reward of +13 over 100 episodes after 499 episodes. In the next section, I will explain the learning algorithm used. 

## Learning Algorithm
The Deep Q-Learning algorithm used in the project consists of two main features: **experienced replay** and **gradual Q-target updates**.

In general, the algorithm is set up by initializing two identical Q-networks, for current and targeted Q-network weights, as well as a replay buffer that will save previously taken steps by the agent, in order to sample them for improved learning.

The agent – as in the discrete learning algorithm – selects an action that either maximizes the action value or is drawn randomly from all possible actions based on the underlying epsilon-greedy policy and a given state.

At each time step, i.e. with each action taken, the agent updates the replay buffer with the values for the current state, reward, action, next state, and whether the episode has terminated.
I have chosen to draw past replays of the agent every 4th action – there are 4 actions in total – and update the targeted Q-learning network based on the sampled experiences.

By updating the target Q-learning network, I mean 
1. selecting the current Q-values from state-action-pairs;
2. choosing the largest Q-values for the next states from the target network;
3. calculating the TD-error;
4. minimizing the loss computed as the mean squared difference between current Q-values and TD-error; and
5. gradually updating the weights on the target network with an small interpolation parameter.

The Q-learning network wouldn't be a network if it doesn't contain a neural network. The neural network consists of 1 input layer, 1 hidden layer and 1 output layer. All layers are fully connected linear layers and map the observation space (states) to action space (actions). The network takes an input of 37 and expands the network to 128 nodes, then contracts to 64 nodes before returning 4 nodes, one for each action. Between each layer there is also a ReLU activation function.

### Parameter Selection
```
NETWORK PARAMETERS
==================
STATE_SIZE = 37         # agent-environment observation space
ACTION_SIZE = 4         # agent's possible action
FC1_UNITS = 128         # first fully-connected layer of network
FC2_UNITS = 64          # second fully-connected layer of network
FC3_UNITS = 64          # third fully-connected layer of network (dueling dqn only)

AGENT PARAMETERS
================
BUFFER_SIZE = int(1e5)  # size of the memory buffer
BATCH_SIZE = 64         # sample batch size
GAMMA = 0.9             # discount rate for future rewards
TAU = 1e-3              # interpolation factor for soft update of target network
LR = 5e-4               # learning rate
UPDATE_EVERY = 4        # after how many steps the network updates

TRAINING PARAMETERS
===================
n_episodes=2000         # max number of episodes to train agent
max_t=1000              # max number of steps agent is taking per episode
eps_start=1.0           # start value of epsilon for epsilon-greedy policy
eps_end=0.01            # terminal value of epsilon
eps_decay=0.995         # discount rate of epsilon for each episode
```

## Training with DQN
The agent is trained with the previously described DQN over 2000 episodes with max. 1000 actions per episode. The DQN employs an epsilon-greedy policy to select appropriate actions, I have chosen to decrease epsilon steadily from 1.0 to 0.01 with every episode by a rate of 0.995.

Below you can see snippet an **untrained** agent take 200 actions:

![untrained][image1]
*The agent is frantically spinning around itself caused by taking actions at random.*

Compared with a **trained** agent: 

![trained][image2]
*The agent steers "consciously" towards the yellow bananas while avoiding the blue ones.*

![scores][image3]
*The distribution cumulative reward per episode shows a clear trendline.*

![episodes count][image4]
*The scores / 100 episodes until goal was achived.*


Even though the game is virtually infinite with bananas dropping from the sky at random, the area of the game is limited and the blue bananas are not disappearing (unless collected). Therefore, the currently trained algorithm tops out at an average cumulative reward of +14 over 100 episodes. 

## Improvements
Further improvements to the algorithm include 
- double Q-learning,
- dueling DQN,
- prioritized experience replay, and
- hyperparameter optimization.

### Double DQN
Reference: [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)

Please refer to agent.py:111 for the implementation of double DQN algorithm, which chooses a next best action based on the local Q-network and evaluates the action values with the target Q-network. Therefore, avoiding to overestimate the action values.

Surprisingly, it actually took longer to reach the goal of +13 over 100 episodes (480 episodes). See graph below.

![doubledqn][image5]
*Scores from Double DQN algorithm*

### Dueling DQN
Reference: [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)

Please refer to model.py:35 for my implementation of the Dueling Q-Network where I split up the value of a state from the advantage of each action in the state. This trains the network to focus on which states are valuable and which are not.

It took 458 episodes to reach the goal of +13 over 100 episodes. See graph below.

![dueldqn][image6]
*Scores from Dueling DQN algorithm* 

### Dueling Double DQN
The combindation of a dueling network and a double DQN improves the agent's training process even further. Taking merely, 396 episodes to train.

![duelddqn][image7]
*Scores from Dueling Double DQN algorithm*

### Prioritized Experience Replay
Reference: [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)

To be implemented...


## Conclusion
The Deep Q-Network learning algorithm is a huge step towards improving reinforcement learning algorithms. While this project relies on a state parameters such as the agent's degree of rotation and 36 others. For video games, relying solely on the input of pixels (the same input that a real player would have) is the highest of challenges. 
There is an extension of the project, which I will pursue as well and test out. This will involve making changes mainly to the underlying neural network, which may use convolutions instead of fully connected layers to make sense of the pixels in an abstract fashion.

