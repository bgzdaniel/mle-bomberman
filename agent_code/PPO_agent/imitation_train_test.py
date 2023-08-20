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
from models_test import state_to_features
from models_test import Agent
import matplotlib.pyplot as plt


# Hyper parameters
BATCH_SIZE = 10
ALPHA = 0.00003
N_EPOCHS = 20
N = 50
N_GAMES = 10000
figure_file = 'C:\\Users\\Maximilian\\Desktop\\bomberman_rl\\agent_code\\PPO_agent\\imitation_scores.png'


# Events
# Define the actions and their corresponding indices
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
n_actions = len(ACTIONS)
input_dims = (7, 17, 17)  # Define the input dimensions for the actor network    

class DotDict:
    def __init__(self, dictionary):
        self.__dict__ = dictionary
        

rounds = 0
#Initialize agent
agent = Agent(n_actions = n_actions, batch_size = BATCH_SIZE, 
              alpha = ALPHA, input_dims = input_dims, n_epochs = N_EPOCHS)
agent.load_models() #load prevous model

#wird jeden step aufgerufen
def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str], before_action: str, same_action_counter: int):
    new_game_state = {
        'round': new_game_state['round'],
        'step': new_game_state['step'],
        'field': np.array(new_game_state['field']),
        'self': new_game_state['self'],
        #'others': [tuple(agent_info) for agent_info in new_game_state['others']],
        'others': new_game_state['others'],
        #'bombs': [tuple(bomb_info) for bomb_info in new_game_state['bombs']],
        'bombs': new_game_state['bombs'],
        'coins': new_game_state['coins'],
        'user_input': new_game_state['user_input'],
        'explosion_map': np.array(new_game_state['explosion_map']),
    }
    old_game_state = {
        'round': old_game_state['round'],
        'step': old_game_state['step'],
        'field': np.array(old_game_state['field']),
        'self': old_game_state['self'],
        #'others': [tuple(agent_info) for agent_info in new_game_state['others']],
        'others': old_game_state['others'],
        #'bombs': [tuple(bomb_info) for bomb_info in new_game_state['bombs']],
        'bombs': old_game_state['bombs'],
        'coins': old_game_state['coins'],
        'user_input': old_game_state['user_input'],
        'explosion_map': np.array(old_game_state['explosion_map']),
    }
    
    
    expert_action = self_action  # Use the expert action from the dataset

    done = False
    reward = 0  # Initialize the reward

    # Check if the agent's action matches the expert action
    action, prob, val = self.give_back_all(old_game_state)
    print("expert action:", expert_action)
    print("action:", action)
    print("before action:", before_action)
    action_list.append(action)
    bomb_count = action_list.count("BOMB")
    if expert_action == action:
        reward = reward + 400  # Positive reward for matching the expert action
    else:
        reward = reward -600  # Negative reward for not matching the expert action
    
    if action == before_action:
        same_action_counter += 1
    
    if action == before_action and action != expert_action:
        reward = reward-100
        
    if action != before_action:
        same_action_counter = 0
    if same_action_counter >= 3:
        print("4 in a row")
        reward = reward-1000
    
    window_size = 4
    # Iterate through the list with a sliding window
    for i in range(len(action_list) - window_size + 1):
        window = action_list[i:i+window_size]
        bomb_count = window.count("BOMB")
    
        if bomb_count > 2:
            reward -= 200
    
        if bomb_count < 2 and bomb_count == 1:
            reward = reward + 500
    
    print("same_action_counter: ", same_action_counter)
    before_action = action
    
    score = reward #errechnet basierend uaf den events diesen step den reward
    
    
    
    # Store the expert action, probability, value, reward, and done flag in memory
    self.remember(state_to_features(old_game_state).flatten(), action, prob, val, reward, done)

    # Train the agent if enough steps are stored in memory
    if len(self.step_scores) % N == 0:
        self.learn()
        self.game_iterations += 1
    self.store_step_scores(score)
    
    return same_action_counter, before_action, action_list

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    
    game_score = sum(self.step_scores)/len(self.step_scores)
    self.store_game_scores(game_score)
    self.clear_step_scores()
    

    self.one_game_score = np.mean(self.game_scores)
    
    if len(self.game_scores) >= 5:
        self.avg_score = np.median(self.game_scores[-5:])
    
    if self.avg_score < -500:
        self.load_models()

    if self.avg_score > self.min_score:
        self.best_score = self.avg_score
        self.set_min_score(self.best_score)
        print("avg_score: ", self.avg_score)
        print("min_score: ", self.min_score)
        self.save_models()
        if len(self.game_scores)>5:
            saved_model_indexes.append(len(self.game_scores) - 5)  # Add the index of the saved model
        else:
            saved_model_indexes.append(len(self.game_scores) - 1)  # Add the index of the saved model

            
    x = [i+1 for i in range(len(self.game_scores))]
    plot_learning_curve(x, self.game_scores, figure_file, saved_model_indexes)

            
        




def plot_learning_curve(x, scores, figure_file, saved_model_indexes):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-10):(i+1)])
    
    plt.plot(x, running_avg)
    plt.title('Reward average of previous 10 games')
    
    for idx in saved_model_indexes:
        plt.annotate('Saved Model', (x[idx], running_avg[idx]), 
                     textcoords="offset points", xytext=(0,10), ha='center', fontsize=8,
                     arrowprops=dict(arrowstyle='->', color='r'))
    
    plt.savefig(figure_file)



# Load the JSON data from multiple files
train_data_directory = 'C:\\Users\\Maximilian\\Desktop\\bomberman_rl\\Dataset10000'
train_data_files = sorted([f for f in os.listdir(train_data_directory) if f.endswith('.json')])


# Create lists to store losses and training steps
games = 0

# Initialize a dictionary to store missed step counts for each file
missed_steps_count = {}
saved_model_indexes = []


while True:
    for data_file in train_data_files:
        
        # Randomly select a file from the test set for validation
        random_test_file = random.choice(train_data_files)
        with open(os.path.join(train_data_directory, random_test_file), 'r') as json_file:
            data = json.load(json_file)
    
        # Initialize missed step count for the current file
        missed = 0
        games += 1
        #print(f"round: {data_file}")
        print(games)
        
        before_action = "null"
        same_action_counter = 0
        action_list = []
        
    
        train_loss_this_file = []  # List to store mean losses for each tested file
        
        for i, step_entry in enumerate(data):
            
        
            state_data = step_entry.get("state")
            #print(state_data)
            action_str = step_entry.get("action")
            events = step_entry.get("events")
            #print(action_str)
            next_state_data = 0

            # Get the next step_entry if it exists
            if i < len(data) - 1:
                next_step_entry = data[i + 1]
                next_state_data = next_step_entry.get("state")
                # Now you have the "next_state_data" for the next step
        
            if i >= 10:  # Stop iterating after the first 5 steps
                end_of_round(agent, state_data, action_str, events)
                break

            #print (next_state_data)
            #if next_state_data is None or 0 and action_str is None:
             #   continue
            if action_str is None:
                missed += 1
                if missed>10:
                    end_of_round(agent, state_data, action_str, events)
                    break
                continue
            
            if next_state_data is None or not next_state_data:
                end_of_round(agent, state_data, action_str, events)
                break
            
            if action_str is None:
                continue
            
            #if next_state_data is None or action_str is None:
             #   agent.end_of_round(state_data, action_str)
              #  continue
            
            same_action_counter, before_action, action_list = game_events_occurred(agent, state_data, action_str, next_state_data, events, before_action, same_action_counter)
            
         # If there are no more JSON files, restart the loop
    if len(train_data_files) == 0:
        train_data_files = sorted([f for f in os.listdir(train_data_directory) if f.endswith('.json')])