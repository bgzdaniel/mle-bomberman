import os
import pickle
import random
import torch

import numpy as np
from .models import Agent

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
    
    # Hyper parameters
    BATCH_SIZE = 5
    ALPHA = 0.0003
    N_EPOCHS = 10
    N_GAMES = 10000
    figure_file = 'C:\\Users\\Maximilian\\Desktop\\bomberman_rl\\agent_code\\PPO_agent\\scores.png'
    
    ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
    n_actions = len(ACTIONS)
    input_dims = (7, 17, 17)
    
    

    self.agent = Agent(n_actions = n_actions, batch_size = BATCH_SIZE, 
                    alpha = ALPHA, input_dims = input_dims, n_epochs = N_EPOCHS)
    #self.agent.save_models()
    self.agent.load_models()
    
    if self.train == True:        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        self.device = torch.device("cpu")
    
    

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    
    action = self.agent.give_back_action (game_state, self.train)
    
    return action

