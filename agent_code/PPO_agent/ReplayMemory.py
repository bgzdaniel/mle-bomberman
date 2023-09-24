import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import matplotlib.pyplot as plt

class ReplayMemory:
    def __init__(self, batch_size):
        
        self.probability = []
        self.finish = []
        self.value = []
        self.actions = []
        self.rewards = []
        self.states = []
    
        
        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        
        if n_states < self.batch_size:
            return np.array(self.states), np.array(self.actions), np.array(self.probability), np.array(self.value), np.array(self.rewards), np.array(self.finish), [np.arange(n_states)]
        #print(n_states)
        #print(np.array(self.states))
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states), np.array(self.actions), np.array(self.probability), np.array(self.values), np.array(self.rewards), np.array(self.finish), batches

    def store_memory(self, state, action, probability, values, reward, finish):
        self.states.append(state)
        self.actions.append(action)
        self.probability.append(probability)
        self.values.append(values)
        self.rewards.append(reward)
        self.finish.append(finish)

    def clear_memory(self):
        self.states = []
        self.values = []
        self.finish = []
        self.probability = []
        self.actions = []
        self.rewards = []
        
       