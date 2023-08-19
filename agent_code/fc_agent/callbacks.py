import os
import pickle
import random

import numpy as np
import torch
from torch import nn

from .rule_based_callbacks import rb_act, rb_setup
        

class DqnNet(nn.Module):
    
    def __init__(self, outer_self):
        super().__init__()
        hidden_size = 64 # 128
        self.fc1 = nn.Linear(outer_self.input_channels, hidden_size)
        self.relu1 = nn.ReLU()
        #self.fc2 = nn.Linear(hidden_size, hidden_size)
        #self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_size, len(outer_self.actions))

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        #x = self.fc2(x)
        #x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return x


def setup(self):
    self.actions = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

    self.epsilon = 1
    self.epsilon_end = 0.05
    self.epsilon_decay = 0.9975

    self.field_shape = (17, 17)
    self.input_channels = 17*17

    self.conv_block_size = 1
    self.depth = 8
    self.init_channels = 32

    self.field_dim = 0
    self.bombs_dim = 1
    self.bombs_rad_dim = 2
    self.explosion_dim = 3
    self.coins_dim = 4
    self.myself_dim = 5
    self.other_dim = 6

    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {self.device}\n')
    
    if self.train or not os.path.isfile("fc_agent_model.pth"): 
        self.policy_net = DqnNet(self).to(self.device)
    else:
        self.target_net = torch.load("fc_agent_model.pth", map_location=self.device)
        self.policy_net = torch.load("fc_agent_model.pth", map_location=self.device)
        
    


    rb_setup(self)


def act(self, game_state: dict) -> str:
    features = state_to_features(self, game_state)
    if self.train == True:
        rand = random.random()
        if rand <= self.epsilon:
            action = random.randint(0, len(self.actions)-1)
            if self.epsilon > self.epsilon_end:
                self.epsilon *= self.epsilon_decay
            else:
                self.epsilon = self.epsilon_end
            #select action from rule based agent
            #action = rb_act(self, game_state)
            #if action is None:
            #    action = 'WAIT'
            #return action
        else:
            with torch.no_grad():
                features = torch.from_numpy(features).to(self.device)[None]
                predictions = self.policy_net(features)
            action = torch.argmax(predictions).item()
    else:
        with torch.no_grad():
            features = torch.from_numpy(features).to(self.device)[None]
            predictions = self.policy_net(features)
            action = torch.argmax(predictions).item()
    return self.actions[action]


def get_bomb_rad_dict(game_state):
    bombs = {coords: timer for coords, timer in game_state["bombs"]}
    for coords, timer in game_state["bombs"]:
        for i in range(1, 3+1):
            x = coords[0]
            y = coords[1]
            bombradius = [(x, y-i), (x, y+i), (x-i, y), (x+i, y)]
            for bombrad_coord in bombradius:
                if bombrad_coord in bombs:
                    bombs[bombrad_coord] = min(timer, bombs[bombrad_coord])
                else:
                    bombs[bombrad_coord] = timer
    return bombs


def state_to_features(self, game_state: dict):
    if game_state is None:
        return None

    # for testing
    # game_state["bombs"] = [((1, 2), 3), ((3, 4), 5)]
    # game_state["coins"] = [(0, 1), (1, 2), (2, 3)]
    # game_state["others"] = [("", 0, True, (5, 6)), ("", 0, False, (5, 5))]


    field = game_state["field"]

    field = game_state['field']
    
    # map all entities onto the field
    for bomb in game_state['bombs']:
        # for now, we ignore that the bomb has a timer
        field[bomb[0]] = 2

    for coin in game_state['coins']:
        field[coin] = 3

    for opponent in game_state['others']:
        field[opponent[3]] = 4

    field[game_state['self'][3]] = 5    
    
    # convert field to numpy array and flatten array
    field = np.array(field).flatten().astype(np.float32)
    return field