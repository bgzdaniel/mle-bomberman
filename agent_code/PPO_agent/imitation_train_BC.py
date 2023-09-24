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


# Hyper parameters
BATCH_SIZE = 10
ALPHA = 0.004
N_EPOCHS = 20
N = 50
N_GAMES = 10000
figure_file = 'C:\\Users\\Maximilian\\Desktop\\bomberman_rl\\agent_code\\PPO_agent\\imitation_scores.png'


# Events
# Define the actions and their corresponding indices
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
n_actions = len(ACTIONS)
input_dims = (7, 17, 17)  # 7,17, 17 for the whole input 5,1,1 for the small input    

#small class to translate the json file back to the orginial form
class DotDict:
    def __init__(self, dictionary):
        self.__dict__ = dictionary
        

#Initialize agent
agent = Agent(n_actions = n_actions, batch_size = BATCH_SIZE, 
              alpha = ALPHA, input_dims = input_dims, n_epochs = N_EPOCHS)
agent.load_models() #load previous model

def plot_loss(loss, filename='C:\\Users\\Maximilian\\Desktop\\bomberman_rl\\agent_code\\PPO_agent\\bc_loss_plot.png', window_size=20):
    """
    Plots the average loss values over the last 'window_size' iterations.

    Parameters:
    loss (list): List of loss values to be plotted.
    window_size (int): The number of iterations to use for calculating the moving average.
    """
    if len(loss) < window_size:
        # Not enough data points to calculate moving average, plot all points.
        iterations = list(range(1, len(loss) + 1))
        avg_loss = loss
    else:
        # Calculate the moving average
        iterations = list(range(window_size, len(loss) + 1))
        avg_loss = [sum(loss[i - window_size:i]) / window_size for i in range(window_size, len(loss) + 1)]

    # Create a line plot of average loss vs. iterations
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, avg_loss, marker='o', linestyle='-')
    plt.title('Moving Average Loss vs. Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Average Loss')
    plt.grid(True)
    plt.savefig(filename)

def plot_loss200(loss, filename='C:\\Users\\Maximilian\\Desktop\\bomberman_rl\\agent_code\\PPO_agent\\bc_loss_plot_200.png', window_size=200):
    if len(loss) < window_size:
        iterations = list(range(1, len(loss) + 1))
        avg_loss = loss
    else:
        iterations = list(range(window_size, len(loss) + 1))
        avg_loss = [sum(loss[i - window_size:i]) / window_size for i in range(window_size, len(loss) + 1)]

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, avg_loss, marker='o', linestyle='-')
    plt.title('Moving Average Loss vs. Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Average Loss')
    plt.grid(True)
    plt.savefig(filename)
    
def plot_loss500(loss, filename='C:\\Users\\Maximilian\\Desktop\\bomberman_rl\\agent_code\\PPO_agent\\bc_loss_plot_500.png', window_size=500):
    if len(loss) < window_size:
        iterations = list(range(1, len(loss) + 1))
        avg_loss = loss
    else:
        iterations = list(range(window_size, len(loss) + 1))
        avg_loss = [sum(loss[i - window_size:i]) / window_size for i in range(window_size, len(loss) + 1)]

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, avg_loss, marker='o', linestyle='-')
    plt.title('Moving Average Loss vs. Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Average Loss')
    plt.grid(True)
    plt.savefig(filename)


#Load the example expert data generated from 10000 Games always saving the actoin and state of eaxh step of the agent which won the game
# Load the JSON data from multiple files
train_data_directory = 'C:\\Users\\Maximilian\\Desktop\\bomberman_rl\\Dataset10000'
train_data_files = sorted([f for f in os.listdir(train_data_directory) if f.endswith('.json')])

# Initialize a dictionary to store missed step counts for each file. Reason is that in the expert data, the agent who won the game is not necessarily the one which survived the longest, so there is saved games where after some point the "action" is always "none".
missed_steps_count = {}
saved_model_indexes = []
loss = []

#prevoius code for Bhavioural cloning. Works  just as well as the newer one just has problems with the necessray memory to run it for a lot of games. Crashes around 2500 games.
"""
i = 0
while True:
    for data_file in train_data_files:
        
        print("Game:", i)
        i += 1
        if i%250 == 0:
            agent.save_models()
            
            
            
        # Randomly select a file from the test set for validation
        random_test_file = random.choice(train_data_files)
        with open(os.path.join(train_data_directory, random_test_file), 'r') as json_file:
            data = json.load(json_file)
            print(json_file)
        avg_bc_loss = agent.behavior_cloning(data, bc_learning_rate=0.0005, batch_size=1)  #gut gelaufen mit 0.0002 und bacth size 2
        loss.append(avg_bc_loss)
        plot_loss(loss)
        plot_loss200(loss)
        plot_loss500(loss)
            
         # If there are no more JSON files, restart the loop
    if len(train_data_files) == 0:
        train_data_files = sorted([f for f in os.listdir(train_data_directory) if f.endswith('.json')])
        
agent.save_models()"""

#Actual Behavioural Cloning part:
#Function to process the Json files from the expert data
def process_data_file(file_path):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
        #print(json_file)
        #behavior_cloning is a functioni from the Agent class.
        avg_bc_loss = agent.behavior_cloning(data, bc_learning_rate=0.0005, batch_size=1)
    return avg_bc_loss

#Loop to loop trough all expert files:
while True:
    for i, data_file in enumerate(train_data_files):
        print("Game:", i)
        
        #save the model every 250 expert games
        if i % 250 == 0:
            agent.save_models()
            
        # Randomly select a file from the test set for validation. Theorethically does not make a difference in our test set. But in case a sorted test set is used one should use random selection.
        random_test_file = random.choice(train_data_files)
        avg_bc_loss = process_data_file(os.path.join(train_data_directory, random_test_file))
        #Just for plotting the loss
        loss.append(avg_bc_loss)
        plot_loss(loss)
        plot_loss200(loss)
        plot_loss500(loss)
        print("Average BC Loss:", avg_bc_loss)

        
agent.save_models()
