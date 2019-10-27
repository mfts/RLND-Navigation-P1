import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    ''' Regular DQN with 1 hidden fully connected layer '''
    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=64):
        '''
        -----------------------------------
        Parameters
        
        state_size:  # of states
        action_size: # of actions
        seed:        random seed
        fc1_units:   # of nodes in first hidden layer
        fc2_units:   # of nodes in second hidden layer
        -----------------------------------
        '''
        super().__init__()
        self.seed = torch.manual_seed(seed)
        
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, fc1_units),
            nn.ReLU(),
            nn.Linear(fc1_units, fc2_units),
            nn.ReLU(),
            nn.Linear(fc2_units, action_size)
        )
            
    
    def forward(self, state):
        ''' Build a deep neural network that has 3 fully connected layers from state_size to action_size '''
        return self.feature_layer(state)
    
class DuelQNetwork(nn.Module):
    ''' Dueling DQN with 2 hidden fully connected layers '''
    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=64, fc3_units=64):
        '''
        -----------------------------------
        Parameters
        
        state_size:  # of states
        action_size: # of actions
        seed:        random seed
        fc1_units:   # of nodes in first hidden layer
        fc2_units:   # of nodes in second hidden layer
        -----------------------------------
        '''
        super().__init__()
        self.seed = torch.manual_seed(seed)
        
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, fc1_units),
            nn.ReLU(),
            nn.Linear(fc1_units, fc2_units),
            nn.ReLU()
        )
        
        self.value_stream = nn.Sequential(
            nn.Linear(fc2_units, fc3_units),
            nn.ReLU(),
            nn.Linear(fc3_units, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(fc2_units, fc3_units),
            nn.ReLU(),
            nn.Linear(fc3_units, action_size)
        )
    
    def forward(self, state):
        features = self.feature_layer(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        return values + (advantages - advantages.mean())