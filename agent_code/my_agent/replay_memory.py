import pickle
import random

class ReplayMemory:

    def __init__(self, capacity, batch_size):
        self.file_name = "replay_memory.pkl"
        self.capacity = capacity
        self.batch_size = batch_size

        try:
            with open(self.file_name, "rb") as file:
                self.memory = pickle.load(file)
        except FileNotFoundError:
            self.memory = []

    def push(self, transition):
        """Adds a transition to memory"""
        self.memory += transition
        if len(self.memory) > self.capacity:
            del self.memory[:len(self.memory) - self.capacity]

    def sample(self):
        """Returns a random sample of transitions"""
        return random.sample(self.memory, self.batch_size)
    
    def save(self):
        """Saves the memory to a file"""
        with open(self.file_name, "wb") as file:
            pickle.dump(self.memory, file)

    def size(self):
        return len(self.memory)