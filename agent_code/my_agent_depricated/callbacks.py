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
    self.input_channels = 5

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
    
    if self.train: # or not os.path.isfile("fc_agent_model.pth"): 
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
            self.logger.debug(f'Action: {self.actions[action]}')
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

def bomb_is_lethal(agent_position, bomb_position):
    if agent_position[0] != bomb_position[0] and agent_position[1] != bomb_position[1]:
        return False 
    if agent_position[0] == bomb_position[0] and np.abs(agent_position[1] - bomb_position[1]) <= 3:
        return True
    if agent_position[1] == bomb_position[1] and np.abs(agent_position[0] - bomb_position[0]) <= 3:
        return True

def state_to_features(self, game_state: dict):
    if game_state is None:
        return None

    agent_position = np.array(game_state['self'][3])

    go_north = 0
    go_south = 0
    go_west = 0
    go_east = 0
    drop_bomb = 0

    # where can the agent go?
    field = game_state['field']
    
    north = (agent_position[0], agent_position[1]-1)
    south = (agent_position[0], agent_position[1]+1)
    west = (agent_position[0]-1, agent_position[1])
    east = (agent_position[0]+1, agent_position[1])

    if field[north] == 0:
        go_north = 1
    if field[south] == 0:
        go_south = 1
    if field[west] == 0:
        go_west = 1
    if field[east] == 0:
        go_east = 1

    # should the agent drop a bomb? 
    agent_surroundings = np.array([field[north], field[south], field[west], field[east]])
    # drop bomb if 2 or more crates surround agent
    if np.count_nonzero(agent_surroundings == 1) >= 2:
        drop_bomb = 1
    # drop bomb if there's 1 crate and 2 walls around agent
    elif np.count_nonzero(agent_surroundings == 1) == 1 and np.count_nonzero(agent_surroundings == -1) == 2:
        drop_bomb = 1

    # if there's a bomb close to the agent, it should only see directions away from the bomb
    bombs = game_state["bombs"]
    if len(bombs) > 0:
        bomb_distances = np.array([np.linalg.norm(np.array(bomb[0]) - np.array(agent_position)) for bomb in bombs])
        closest_bomb_index = np.argmin(bomb_distances)
        closest_bomb = bombs[closest_bomb_index]
        if bomb_is_lethal(agent_position, closest_bomb[0]):
            drop_bomb = 0
            escape_north = 0
            escape_south = 0
            escape_west = 0
            escape_east = 0
            escape_found = False
            if go_north:
                if not bomb_is_lethal(agent_position+[0,1], closest_bomb[0]):
                    escape_north = 1
                    escape_found = True
            if go_south:
                if not bomb_is_lethal(agent_position+[0,-1], closest_bomb[0]) and not escape_found:
                    escape_south = 1
                    escape_found = True
            if go_west:
                if not bomb_is_lethal(agent_position+[1,0], closest_bomb[0]) and not escape_found:
                    escape_west = 1
                    escape_found = True
            if go_east:
                if not bomb_is_lethal(agent_position+[-1,0], closest_bomb[0]) and not escape_found:
                    escape_east = 1
                    escape_found = True

            if escape_found:
                go_north = escape_north
                go_south = escape_south
                go_west = escape_west
                go_east = escape_east
            else:
                # only allow directions away from the bomb (when agent on bomb, just go anywhere valid)
                if agent_position != closest_bomb:
                    bomb_direction = np.array(agent_position) - np.array(closest_bomb[0])
                    if bomb_direction[1] < 0:
                        go_north = 0
                    if bomb_direction[1] > 0:
                        go_south = 0
                    if bomb_direction[0] < 0:
                        go_west = 0
                    if bomb_direction[0] > 0:
                        go_east = 0

        # now check if moving might get agent into bomb radius
        if bomb_distances[closest_bomb_index] <= 5: # not sure if 5 is correct
            if go_north and bomb_is_lethal(agent_position+[0,1], closest_bomb[0]):
                go_north = 0
            if go_south and bomb_is_lethal(agent_position+[0,-1], closest_bomb[0]):
                go_south = 0
            if go_west and bomb_is_lethal(agent_position+[-1,0], closest_bomb[0]):
                go_west = 0
            if go_east and bomb_is_lethal(agent_position+[1,0], closest_bomb[0]):
                go_east = 0

    # add bias for coins
    # add bomb cool-down
    #TODO

    self.logger.debug(f'X{go_north}X')
    self.logger.debug(f'{go_west}{drop_bomb}{go_east}')
    self.logger.debug(f'X{go_south}X')

    field = np.array(game_state['field'], dtype=object)
    for bomb in game_state['bombs']:
        field[bomb[0]] = 'B'

    for coin in game_state['coins']:
        field[coin] = 'C'

    for opponent in game_state['others']:
        field[opponent[3]] = 'X'

    field[game_state['self'][3]] = 'A'  

    # replace all zeros in field with ' '
    field =  np.where(field == -1, '%', field)
    field =  np.where(field == 1, 'X', field)
    field = np.where(field == 0, ' ', field)
    self.logger.debug(f'\n{field.T}')

    return np.array([go_north, go_south, go_west, go_east, drop_bomb]).flatten().astype(np.float32)