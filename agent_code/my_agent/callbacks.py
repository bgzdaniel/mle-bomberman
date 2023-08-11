import os
import pickle
import random
import torch

import numpy as np
from .model import Model
from .hyperparameters import hp

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        #weights = np.random.rand(len(ACTIONS))
        #self.model = weights / weights.sum()

        # here we do the second step from the paper 'Initialize action-value function Q w/ random weights'
        self.q_network = Model(n_features=17*17, n_actions=len(ACTIONS))
        self.target_network = Model(n_features=17*17, n_actions=len(ACTIONS))
    else:
        pass # figure out how to do this later
        #self.logger.info("Loading model from saved state.")
        #with open("my-saved-model.pt", "rb") as file:
        #    self.model = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    if self.train and random.random() < hp.epsilon:
        self.logger.debug("Choosing action purely at random.")
        # idea: use decision of rule-based agent here
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    self.logger.debug("Querying model for action.")
    action = self.q_network(state_to_features(game_state))
    action = ACTIONS[torch.argmax(action).item()]
    self.logger.debug(f"Action Prediction by Q-Network: {action}")
    return action


def state_to_features(game_state: dict) -> torch.tensor:
    """ Maps the game state to a 17x17 grid that contains values that represent the game field.
    The following values are used:
    wall: -1
    free tile: 0
    crate: 1
    bomb: 2
    coin: 3
    opponent: 4
    agent: 5
    """

    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

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
    
    return torch.tensor(field, dtype=torch.float32).flatten()