from collections import namedtuple, deque
import random

import torch
from torch import nn
from torch import optim
import numpy as np

from typing import List

import events as e
from .callbacks import state_to_features, DqnNet
from .utility import DataCollector

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
TRANSITION_HISTORY_SIZE = int(1e6)
DISCOUNT = 0.9

MOVE_ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT']

def sample(self, batch_size):
    weights = []
    for replay in self.transitions:
        weight = 1 if type(replay.next_state) is np.ndarray else 5
        weights.append(weight)
    return random.choices(self.transitions, weights=weights, k=batch_size)


def setup_training(self):
    self.batch_size = 32
    self.target_net = DqnNet(self).to(self.device)
    self.target_net.load_state_dict(self.policy_net.state_dict())
    self.loss_function = nn.SmoothL1Loss()  # Huber Loss as proposed by the paper
    self.optimizer = optim.Adam(self.policy_net.parameters())
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.steps_per_copy = 2500
    self.train_iter = 0
    
    # for logging
    self.round = 0
    self.loss_per_step = []
    self.reward_per_step = []
    self.invalid_actions_per_round = 0
    self.bombs_dropped_per_round = 0
    self.weights_copied_iter = 0

    self.data_collector = DataCollector("score_per_round.txt")
    self.data_collector.initialize()


def get_reward(self, events, old_features):
    reward = 0.0

    if e.INVALID_ACTION in events:
        reward -= 1
    if e.WAITED in events and not (sum(old_features[0:3]) == 0):
        reward -= 0.5
    if e.BOMB_DROPPED in events and old_features[4] == 1:
        reward += 1
    if old_features[4] == 1 and not e.BOMB_DROPPED in events:
        reward -= 1
    if e.MOVED_UP in events and old_features[0] == 1:
        reward += 0.5
    if e.MOVED_DOWN in events and old_features[1] == 1:
        reward += 0.5
    if e.MOVED_LEFT in events and old_features[2] == 1:
        reward += 0.5
    if e.MOVED_RIGHT in events and old_features[3] == 1:
        reward += 0.5

    self.logger.debug(f'Reward: {reward}')
    return reward

def do_every_step(self, old_features, new_features, events: List[str]):
    pass


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    old_features = state_to_features(self, old_game_state)
    new_features = state_to_features(self, new_game_state)

    reward = get_reward(self, events, old_features)
    #print(reward)
    self.reward_per_step.append(reward)

    if e.INVALID_ACTION in events:
        self.invalid_actions_per_round += 1
    if e.BOMB_DROPPED in events:
        self.bombs_dropped_per_round += 1

    self.transitions.append(Transition(old_features, self.actions.index(self_action), new_features, reward))
    
    update_params(self)

    # copy weights to target net after n steps
    if self.train_iter % self.steps_per_copy == 0 and self.train_iter != 0:
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.weights_copied_iter += 1
        self.logger.debug(f"weights copied to target net! ({self.weights_copied_iter} times)\n")

    # increase batch size after every n steps for dampening of fluctuations
    # and faster convergence instead of decaying learning rate (https://arxiv.org/abs/1711.00489)
    if (self.train_iter % (self.steps_per_copy * 10) == 0) and (self.batch_size < 512) and self.train_iter != 0:
        self.batch_size *= 2
        self.logger.debug(f"batch size increased to {self.batch_size}!\n")

    self.train_iter += 1

    self.logger.debug(f"Total Reward: {reward}")
    

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):

    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    last_features = state_to_features(self, last_game_state)

    reward = get_reward(self, events, last_features)
    self.reward_per_step.append(reward)

    if e.INVALID_ACTION in events:
        self.invalid_actions_per_round += 1
    if e.BOMB_DROPPED in events:
        self.bombs_dropped_per_round += 1

    self.transitions.append(Transition(last_features, self.actions.index(last_action), None, reward))

    update_params(self)

    # copy weights to target net after n steps
    if self.train_iter % self.steps_per_copy == 0 and self.train_iter != 0:
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.weights_copied_iter += 1
        self.logger.debug(f"weights copied to target net! ({self.weights_copied_iter} times)\n")

    # increase batch size after every n steps for dampening of fluctuations
    # and faster convergence instead of decaying learning rate (https://arxiv.org/abs/1711.00489)
    if (self.train_iter % (self.steps_per_copy * 10) == 0) and (self.batch_size < 512) and self.train_iter != 0:
        self.batch_size *= 2
        self.logger.debug(f"batch size increased to {self.batch_size}!\n")

    self.train_iter += 1
    self.round += 1
    avg_invalid_actions_per_step = self.invalid_actions_per_round / last_game_state['step']
    killed_self = e.KILLED_SELF in events

    self.data_collector.write(round=last_game_state['step'], 
                              epsilon=self.epsilon, 
                              score=last_game_state["self"][1], 
                              killed_self=killed_self, 
                              avg_loss_per_step=np.mean(self.loss_per_step), 
                              avg_reward_per_step=np.mean(self.reward_per_step),
                              invalid_actions_per_round=self.invalid_actions_per_round,
                              avg_invalid_actions_per_step=avg_invalid_actions_per_step,
                              dropped_bombs=self.bombs_dropped_per_round)

    self.loss_per_step = []
    self.reward_per_step = []
    self.invalid_actions_per_round = 0
    self.bombs_dropped_per_round = 0

    if self.round % 200:
        torch.save(self.target_net, 'fc_agent_model.pth')

    self.logger.debug(f"Total Reward: {get_reward}")

def update_params(self):
    if len(self.transitions) < self.batch_size:
        return

    replays = sample(self, self.batch_size)

    # calculate predictions
    replays_states = torch.cat([torch.from_numpy(replay.state)[None] for replay in replays]).to(self.device)
    replays_actions = torch.tensor([replay.action for replay in replays]).to(self.device)[:, None]
    predictions = torch.gather(self.policy_net(replays_states), 1, replays_actions)

    # calculate targets
    replays_non_terminal_states = []
    for i, replay in enumerate(replays):
        if type(replay.next_state) is np.ndarray:
            replays_non_terminal_states.append(i)
    replays_non_terminal_states = torch.tensor(replays_non_terminal_states).to(self.device)

    replays_next_states = []
    for replay in replays:
        if type(replay.next_state) is np.ndarray:
            replays_next_states.append(torch.from_numpy(replay.next_state)[None])
    replays_next_states = torch.cat(replays_next_states).to(self.device)

    max_future_actions = torch.zeros(self.batch_size, 1).to(self.device)
    max_future_actions[replays_non_terminal_states, :] = torch.max(self.target_net(replays_next_states), dim=1)[0][:, None]

    replays_rewards = torch.tensor([replay.reward for replay in replays]).to(self.device)[:, None]
    targets = replays_rewards + DISCOUNT * max_future_actions

    # calculate loss, gradients and backpropagate
    loss = self.loss_function(predictions, targets)
    self.loss_per_step.append(loss.item())
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    return loss
