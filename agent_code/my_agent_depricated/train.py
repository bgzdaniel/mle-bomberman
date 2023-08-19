from collections import namedtuple, deque
from torch import nn
from torch import optim
import numpy as np
import torch
from typing import List
import matplotlib.pyplot as plt


import events as e
from .callbacks import state_to_features, ACTIONS
from .replay_memory import ReplayMemory
from .hyperparameters import hp
from .resources import Transition

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)

    # data collection to monitor training progress
    self.round = 0
    self.loss_all_rounds = []
    self.loss_per_round = []
    self.score = []
    self.gradients = []
    
    # initialize replay memory
    self.memory = ReplayMemory(hp.memory_size, hp.batch_size, self.device)

    # initialize loss and optimizer
    self.loss_object = nn.SmoothL1Loss()
    self.loss = 0
    self.optimizer = optim.AdamW(self.q_network.parameters(), lr=1e-4, amsgrad=True)

    

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Idea: Add your own events to hand out rewards
    #if ...:
    #    events.append(PLACEHOLDER_EVENT)
    
    current_transition = Transition(state_to_features(old_game_state),
                                    action_to_number(self_action),
                                    state_to_features(new_game_state),
                                    reward_from_events(self, events))
    self.transitions.append(current_transition)
    if current_transition.state is not None:
        self.memory.push(current_transition)

    # only update every n steps?
    if new_game_state["step"] % hp.update_frequency == 0:
        update(self)

        # store the model in the target network (use weights?)
        q_network_state_dict = self.q_network.state_dict()
        self.target_network.load_state_dict(q_network_state_dict)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    self.loss_all_rounds.append(np.mean(self.loss_per_round))
    self.score.append(last_game_state["self"][1])
    self.round += 1

    if self.round == 500:
        plt.bar(range(len(self.loss_all_rounds)), self.loss_all_rounds, color='blue')
        plt.savefig("loss.png")
        plt.close()

        plt.bar(range(len(self.score)), self.score, color='blue')
        plt.savefig("score.png")
        plt.close()

    

def reward_from_events(self, events: List[str]) -> torch.tensor:
    """
    Rewards agent to navigate field (navigate field, destroy crates, collects coins, don't die)
    """
    game_rewards = {
        e.COIN_COLLECTED: 5,
        e.KILLED_OPPONENT: 5,
        e.CRATE_DESTROYED: 0.5,
        e.KILLED_SELF: -10,
        e.INVALID_ACTION: -1,
        e.WAITED: -0.5,
        e.SURVIVED_ROUND: 0.5,
        e.MOVED_DOWN: 0.1,
        e.MOVED_UP: 0.1,
        e.MOVED_RIGHT: 0.1,
        e.MOVED_LEFT: 0.1,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return torch.tensor([reward_sum])

def action_to_number(action) -> torch.tensor:
    return torch.tensor([ACTIONS.index(action)])


def update(self):
    # in this function, we need to do:
    # sample minibatch from replay memory
    if self.memory.size() < hp.batch_size:
        return
    replay_batch = self.memory.sample()

    # calculate the predicted Q values in minibatch
    predictions = self.q_network(replay_batch.state).gather(1, replay_batch.action)
    targets = replay_batch.reward + hp.discount * self.target_network(replay_batch.next_state).max(1)[0].detach().unsqueeze(1)

    # use Huber loss like suggested in the paper
    self.loss = self.loss_object(predictions, targets)
    self.loss_per_round.append(self.loss.item())
    self.optimizer.zero_grad()
    self.loss.backward()
    

    # log the gradient values
    #for tag, value in self.q_network.named_parameters():
    #    tag = tag.replace('.', '/')
    #    self.gradients.append(mean(value.grad.data.cpu().numpy()))
    self.optimizer.step()

def plot(path):
    path = '/content/mle-bomberman/agent_code/cnn_agent/score_per_round.txt'
    with open(path, 'r') as f:
        data = f.read()

    data = data.split('\n')
    data = [line.split('\t') for line in data]
    data = data[:-1]

    data = [line[3:] for line in data] # ignore iter, round, epsilon

    # get labels from first line
    labels = data[0]

    # get data from other lines
    data = data[1:]

    # create plots
    import matplotlib.pyplot as plt
    import numpy as np
    # create bar plots for each label
    # create a plot with 5 subplots that plot the data
    fig, axs = plt.subplots(5, 1, figsize=(10, 10))
    fig.tight_layout(pad=3.0)


    # add the asubplots to the plot
    for i, label in enumerate(labels):
        axs[i].set_title(label)
        axs[i].set_xlabel('round')
        axs[i].set_ylabel(label)
        # fewer data points on y label for floats
        axs[i].plot(np.array(data)[:, i])

    plt.show()

