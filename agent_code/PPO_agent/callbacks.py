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
    BATCH_SIZE = 2
    ALPHA = 0.0003 #or 0.0003
    N_EPOCHS = 5
    N_GAMES = 10000
    figure_file = 'C:\\Users\\Maximilian\\Desktop\\bomberman_rl\\agent_code\\PPO_agent\\scores.png'
    
    ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
    n_actions = len(ACTIONS)
    input_dims = (7, 17, 17)
    
    

    self.agent = Agent(n_actions = n_actions, batch_size = BATCH_SIZE, 
                    alpha = ALPHA, input_dims = input_dims, n_epochs = N_EPOCHS)
    self.agent.save_models()
    #self.agent.load_models()
    #self.step_counter = 0
    
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
    if self.train == True:
        #Here to controll the eplison decay for specific parts/steps of a whole game
        action, prob, val, epsilon = load_values_from_file('values.pkl')
        #self.step_counter += 1
        #if self.step_counter <= 3:
        #    epsilon = 1
        #if self.step_counter == 4:
        #    epsilon = 1
        #if self.step_counter == 10:
        #    epsilon = 0
        #if action == 25:
        #    self.step_counter = 0
        self.agent.epsilon = epsilon
    
    #action = self.agent.give_back_action (game_state, self.train)
    action, prob, val, epsilon = self.agent.give_back_all(game_state, self.train)
    
    if self.train == True:
        save_values_to_file('values.pkl', action, prob, val, epsilon)
    
    return action



def save_values_to_file(file_path, action, prob, val, epsilon):
    data = {
        'action': action,
        'prob': prob,
        'val': val,
        'epsilon': epsilon
    }
    
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)

def load_values_from_file(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    
    action = data['action']
    prob = data['prob']
    val = data['val']
    epsilon = data['epsilon']
    
    return action, prob, val, epsilon
