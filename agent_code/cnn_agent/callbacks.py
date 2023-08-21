import os
import pickle
import random

import numpy as np
import torch
from torch import nn
        

class DqnNet(nn.Module):
    def __init__(self, outer_self):
        super().__init__()
        self.layers = self._make_layers(outer_self)

    def _make_layers(self, outer_self):
        hw = outer_self.field_shape[0]
        prev_channels = outer_self.input_channels
        layers = []
        for i in range(outer_self.depth):
            if i == 0:
                next_channels = outer_self.init_channels
            else:
                next_channels = prev_channels * 2
            for _ in range(outer_self.conv_block_size):
                layers += [
                    nn.Conv2d(prev_channels, next_channels, 3),
                    nn.ReLU(),
                ]
                prev_channels = next_channels
        flatten_size = ((hw - ((2 * outer_self.conv_block_size) * outer_self.depth))**2) * prev_channels
        layers += [
            nn.Flatten(start_dim=1),
            nn.Linear(flatten_size, len(outer_self.actions)),
        ]
        return nn.ModuleList(layers)

    def forward(self, x):
        for module in self.layers:
            x = module(x)
        return x


def setup(self):
    self.actions = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

    self.epsilon = 1
    self.epsilon_end = 0.05
    self.epsilon_decay = 0.999

    self.field_shape = (17, 17)
    self.input_channels = 7

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

    if self.train == True:        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        self.device = torch.device("cpu")
    
    print('Using device:', self.device)
    print()

    self.policy_net = DqnNet(self).to(self.device)


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

    bombs = np.full(self.field_shape, -1, dtype=np.int32)
    bombs_rad = np.full(self.field_shape, -1, dtype=np.int32)
    if len(game_state["bombs"]) != 0:
        bomb_coords = np.array([coords for coords, _ in game_state["bombs"]])
        bomb_timers = np.array([timer for _, timer in game_state["bombs"]])
        bombs[bomb_coords[:, 0], bomb_coords[:, 1]] = bomb_timers
        bomb_rad_dict = get_bomb_rad_dict(game_state)
        for coords, timer in bomb_rad_dict.items():
            x = coords[0]
            y = coords[1]
            if x >= 0 and y >= 0 and x < self.field_shape[0] and y < self.field_shape[1]:
                if bombs_rad[x, y] != -1:
                    bombs_rad[x, y] = min(bombs_rad[x, y], timer)
                else:
                    bombs_rad[x, y] = timer

    explosion_map = game_state["explosion_map"]

    coins = np.zeros(self.field_shape, dtype=np.int32)
    if len(game_state["coins"]) != 0:
        coin_coords = np.array(game_state["coins"])
        coins[coin_coords[:, 0], coin_coords[:, 1]] = 1

    myself = np.zeros(self.field_shape, dtype=np.int32)
    myself_bomb_action = 1 if game_state["self"][2] == True else -1
    myself[game_state["self"][3][0], game_state["self"][3][1]] = myself_bomb_action

    others = np.zeros(self.field_shape, dtype=np.int32)
    if len(game_state["others"]) != 0:
        others_coords = np.array([coords for _, _, _, coords in game_state["others"]])
        others_bomb_action = np.array([bomb_action for _, _, bomb_action, _ in game_state["others"]])
        others_bomb_action = np.where(others_bomb_action == True, 1, -1)
        others[others_coords[:, 0], others_coords[:, 1]] = others_bomb_action

    channels = [field, bombs, bombs_rad, explosion_map, coins, myself, others]
    #feature_maps = np.stack(channels, axis=0, dtype=np.float32) doesn't work on colab
    feature_maps = np.stack(channels, axis=0).astype(np.float32)
    return feature_maps
