import pickle
import random
import torch

from .resources import Transition
from .hyperparameters import hp

class ReplayMemory:

    def __init__(self, capacity: int, batch_size: int, device: str) -> None:
        self.file_name = "replay_memory.pkl"
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device

        try:
            with open(self.file_name, "rb") as file:
                self.memory = pickle.load(file)
        except FileNotFoundError:
            self.memory = []

    def push(self, transition: Transition) -> None:
        """Adds a transition to memory"""
        assert transition.state.shape == torch.Size([17**2])
        assert transition.next_state.shape == torch.Size([17**2])
        assert transition.action.shape == torch.Size([1])
        assert transition.reward.shape == torch.Size([1])

        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            self.memory.pop(0)

    def sample(self) -> Transition:
        """Returns a Transition of torch batches"""
        sample = random.sample(self.memory, self.batch_size)

        # this probably could be refactored
        data = Transition(state=torch.cat([transition.state for transition in sample]).view(hp.batch_size,-1).to(self.device),
                          action=torch.unsqueeze(torch.tensor([transition.action for transition in sample],device=self.device),1),
                          next_state=torch.cat([transition.next_state for transition in sample]).view(hp.batch_size,-1).to(self.device),
                          reward=torch.unsqueeze(torch.tensor([transition.reward for transition in sample],device=self.device),1))
        assert data.state.shape == torch.Size([hp.batch_size,17**2]), data.state.shape
        assert data.next_state.shape == torch.Size([hp.batch_size,17**2]), data.next_state.shape
        assert data.action.shape == torch.Size([hp.batch_size,1]), data.action.shape
        assert data.reward.shape == torch.Size([hp.batch_size,1]), data.reward.shape
        assert data.state.device == self.device
        assert data.next_state.device == self.device
        assert data.action.device == self.device
        assert data.reward.device == self.device
        return data
    
    def save(self) -> None:
        """Saves the memory to a file"""
        with open(self.file_name, "wb") as file:
            pickle.dump(self.memory, file)

    def size(self) -> int:
        return len(self.memory)