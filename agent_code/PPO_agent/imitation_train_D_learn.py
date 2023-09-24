###### import numpy as np
import matplotlib.pyplot as plt
import os
import random
import json
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
from collections import namedtuple, deque
import pickle
import torch
from typing import List
from models_imitation import state_to_features
from models_imitation import Agent
import matplotlib.pyplot as plt

"""Same Setup as in imitation_train_BC.py"""

# Hyper parameters
BATCH_SIZE = 10
ALPHA = 0.003
N_EPOCHS = 20
N = 50
N_GAMES = 10000
figure_file = 'C:\\Users\\Maximilian\\Desktop\\bomberman_rl\\agent_code\\PPO_agent\\imitation_scores.png'


# Events
# Define the actions and their corresponding indices
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
n_actions = len(ACTIONS)
input_dims = (7, 17, 17)  # Define the input dimensions. 5,1,1 for smaller input

class DotDict:
    def __init__(self, dictionary):
        self.__dict__ = dictionary
        

rounds = 0
#Initialize agent
agent = Agent(n_actions = n_actions, batch_size = BATCH_SIZE, 
              alpha = ALPHA, input_dims = input_dims, n_epochs = N_EPOCHS)
agent.load_models() #load prevous model

#Load the expert test data
# Load the JSON data from multiple files
train_data_directory = 'C:\\Users\\Maximilian\\Desktop\\bomberman_rl\\Dataset10000'
train_data_files = sorted([f for f in os.listdir(train_data_directory) if f.endswith('.json')])


missed_steps_count = {}
saved_model_indexes = []

#A Loop to train the discrminator on the expert test set.
while True:
    for data_file in train_data_files:
        
        # Randomly select a file from the test set for validation
        random_test_file = random.choice(train_data_files)
        with open(os.path.join(train_data_directory, random_test_file), 'r') as json_file:
            data = json.load(json_file)
            
        missed = 0
        next_state_data = 0
        
        #iterate trough every step of the game and create expertaction_state sets using the step and the action the expert took that step and also cretae a action_state set using the game state of that set and the action the agent would do given that state as input
        for i, step_entry in enumerate(data):
            state = state_to_features(step_entry.get("state"))
            game_state = step_entry.get("state")
            action_str = step_entry.get("action")
            
            if i < len(data) - 1:
                next_step_entry = data[i + 1]
                next_state_data = next_step_entry.get("state")
                
            if next_state_data is None or not next_state_data:
                break
                
            if action_str is None:
                missed += 1
                if missed>10:
                    action = ACTIONS.index(action_str)
                    expert_states_actions = np.concatenate((state.flatten(), np.array([action])))
                    break
                continue
                
            if action_str is not None:
                action = ACTIONS.index(action_str)
                expert_states_actions = np.concatenate((state.flatten(), np.array([action])))
                
            #giving back the action the agent would take that step    
            action, _, _ = self.give_back_all(game_state)
            state = state_to_features(game_state)
            agent_states_actions = np.concatenate((state.flatten(), np.array([action])))
    
    
        #learning the discriminator using the expert and agent actions to the correspondding state
        agent.learn_discriminator(expert_states_actions, agent_states_actions)
            
         # If there are no more JSON files, restart the loop
    if len(train_data_files) == 0:
        train_data_files = sorted([f for f in os.listdir(train_data_directory) if f.endswith('.json')])