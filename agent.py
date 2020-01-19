import numpy as np
import random
from collections import namedtuple, deque
import wandb
wandb.init(project="deep-rl")

from model import QNetwork as QNetwork

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import heapq
from itertools import count
tiebreaker = count()


BUFFER_SIZE = int(1e5)  # size of the memory buffer
BATCH_SIZE = 64         # sample batch size
GAMMA = 0.9             # discount rate for future rewards
TAU = 1e-3              # interpolation factor for soft update of target network
LR = 5e-4               # learning rate
UPDATE_EVERY = 4        # after how many steps the network updates

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    
    def __init__(self, state_size, action_size, seed):
        '''
        --------------------------------
        Parameters
        
        state_size:  # of states (observation space)
        action_size: # of actions (action space)
        seed:        seed for random
        '''
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        
        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        
        wandb.init(project="deep-rl")
        wandb.watch(self.qnetwork_local)
        wandb.watch(self.qnetwork_target)
        
        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        
        
        self.t_step = 0
        
    def step(self, state, action, reward, next_state, done):
        '''
        Agent takes next step
        - saves most recent environment event to ReplayBuffer
        - load random sample from memory to agent's q-network
        '''
        self.memory.add(state, action, reward, next_state, done)
        
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                   experiences = self.memory.sample()
                   self.learn(experiences, GAMMA)
        
    def act(self, state, eps=0.2):
        '''
        Agent selects action based on current state and epsilon-greedy policy
        '''
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        
        # Epsilon-greedy policy
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
        
    def learn(self, experiences, gamma):
        '''
        Agent updates network parameters based on experiences (state, action, reward, next_state, done)
        '''
        states, actions, rewards, next_states, dones = experiences
        
        # get current Q
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        # get Qsa_next
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # calculate target with reward and Qsa_next
        Q_targets = rewards + (gamma* Q_targets_next * (1-dones)) 
        
        # calculate loss
        loss = F.mse_loss(Q_expected, Q_targets)
        
        wandb.log({"Loss": loss})
        
        # minimize loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # update target network parameters
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)
    
    def doublelearn(self, experiences, gamma):
        '''
        Implement Double DQN
        Agent updates network parameters based on experiences (state, action, reward, next_state, done)
        '''
        states, actions, rewards, next_states, dones = experiences
        
        # get current Q
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        # get next best action based on local qnetwork
        Q_local_actions_next = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
        # get next action value from target network
        Q_targets_next = self.qnetwork_target(next_states).gather(1, Q_local_actions_next)
        # calculate target with reward and Qsa_next
        Q_targets = rewards + (gamma* Q_targets_next * (1-dones)) 
        
        # calculate loss
        loss = F.mse_loss(Q_expected, Q_targets)
        
        # minimize loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # update target network parameters
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)
      
    def soft_update(self, local_model, target_model, tau):
        '''
        Update target network weights gradually with an interpolation rate of TAU
        '''
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
class ReplayBuffer:
    
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        
    def add(self, state, action, reward, next_state, done):
        ''' add learning experiences to memory '''
        e = self.experience(state,action,reward,next_state,done)
        self.memory.append(e)
        
    def sample(self):
        ''' return random batch of experiences from memory '''
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        return len(self.memory)
    
