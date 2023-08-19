from collections import namedtuple, deque
import random

import torch
from torch import nn
from torch import optim
import numpy as np

import pickle
from typing import List

import events as e
from .callbacks import state_to_features, DqnNet, get_bomb_rad_dict
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
    self.weights_copied_iter = 0
    self.escaped_bombs = 0

    self.data_collector = DataCollector("score_per_round.txt")
    self.data_collector.initialize()

def reward_from_events(self, events: List[str]) -> int:
    total_reward = 0

    game_rewards = {
        e.INVALID_ACTION: -0.5, # invalid actions waste time
        e.WAITED: -0.25, # need for pro-active agent
        e.CRATE_DESTROYED: 2,
        e.COIN_FOUND: 3,
        e.COIN_COLLECTED: 20,
        e.KILLED_OPPONENT: 100,
        e.SURVIVED_ROUND: 200 # note: the agent can only get this if you win the round or live until round 400
    }

    for event in events:
        if event in game_rewards:
            total_reward += game_rewards[event]

    if e.KILLED_SELF in events or e.GOT_KILLED in events:
        total_reward += -125

    self.logger.debug(f"Reward from events: {total_reward}")
    return total_reward

def reward_from_actions(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str], old_features, new_features):
    total_reward = 0

    # get bomb coords and timers for whole radius and player coords
    new_bombs_rad = get_bomb_rad_dict(new_game_state)
    old_bombs_rad = get_bomb_rad_dict(old_game_state)
    new_player_coord = new_game_state["self"][3]
    old_player_coord = old_game_state["self"][3]

    scaling = 5
    # punish agent for being in bomb radius
    if new_player_coord in new_bombs_rad:
        total_reward += ((new_bombs_rad[new_player_coord] - 4) * scaling)
    # reward agent for stepping out of bomb radius
    elif old_player_coord in old_bombs_rad and new_player_coord not in new_bombs_rad:
        self.escaped_bombs += 1
        total_reward += ((old_bombs_rad[old_player_coord] - 4) * scaling) * -1 * 0.5

    self.logger.debug(f"Reward for bombs: {total_reward}")

    """
    # add reward if agents moves away from bomb
    if len(old_game_state["bombs"]) > 0 & len(new_game_state["bombs"]) > 0:
        new_bomb_coords = np.array([bomb[0] for bomb in new_game_state["bombs"]])
        old_bomb_coords = np.array([bomb[0] for bomb in old_game_state["bombs"]])
        new_distance_to_bomb = np.linalg.norm(new_bomb_coords - np.array(new_player_coord)).min()
        old_distance_to_bomb = np.linalg.norm(old_bomb_coords - np.array(old_player_coord)).min()
        if new_distance_to_bomb > old_distance_to_bomb:
            total_reward += 15
            self.logger.debug(f"Reward for bomb escape: {15}")
    """

    # Daniel comment to below: I think this will prevent the agent from stepping into the bomb radius 
    # to get to a coin and still escape the bombs explosion, which would be nice to have

    # # if the agent is the bomb radius it should ignore coins
    # if new_player_coord in new_bombs_rad:
    #     return total_reward

    # reward agent for getting close to nearest coin
    if (self_action in MOVE_ACTIONS) and (e.COIN_COLLECTED not in events) and len(new_game_state['coins']) > 0 and len(old_game_state['coins']) > 0:
        coin_reward = 0
        new_distances = []
        for coin_coord in new_game_state["coins"]:
            new_distances.append(np.linalg.norm(np.array(coin_coord) - np.array(new_player_coord)))
        new_min_distance = np.min(np.array(new_distances))

        old_distances = []
        for coin_coord in old_game_state["coins"]:
            old_distances.append(np.linalg.norm(np.array(coin_coord) - np.array(old_player_coord)))
        old_min_distance = np.min(np.array(old_distances))
        coin_reward += (old_min_distance - new_min_distance) * 0.2
        
        reward_for_coin_proximity = (old_min_distance - new_min_distance) * 0.2
        # weight reward depending on distance to nearest coin
        reward_for_coin_proximity *= 1/(new_min_distance)**2
        coin_reward += reward_for_coin_proximity

        self.logger.debug(f"Reward for coins: {coin_reward}")

        total_reward += coin_reward

    return total_reward

    # TO-DO: reward agent for placing bombs which would hit other players and crates


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    old_features = state_to_features(self, old_game_state)
    new_features = state_to_features(self, new_game_state)

    reward = 0
    reward += reward_from_events(self, events)
    reward += reward_from_actions(self, old_game_state, self_action, new_game_state, events, old_features, new_features)

    reward /= 10
    self.reward_per_step.append(reward)

    if e.INVALID_ACTION in events:
        self.invalid_actions_per_round += 1

    self.transitions.append(Transition(old_features, self.actions.index(self_action), new_features, reward))
    
    loss = update_params(self)

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

    reward = reward_from_events(self, events) / 10

    self.reward_per_step.append(reward)

    if e.INVALID_ACTION in events:
        self.invalid_actions_per_round += 1

    self.transitions.append(Transition(last_features, self.actions.index(last_action), None, reward))

    loss = update_params(self)

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

    self.data_collector.write(train_iter=self.train_iter, 
                              round=self.round, 
                              epsilon=self.epsilon, 
                              score=last_game_state["self"][1], 
                              killed_self=killed_self, 
                              avg_loss_per_step=np.mean(self.loss_per_step), 
                              avg_reward_per_step=np.mean(self.reward_per_step),
                              invalid_actions_per_round=self.invalid_actions_per_round,
                              avg_invalid_actions_per_step=avg_invalid_actions_per_step,
                              escaped_bombs=self.escaped_bombs)

    self.loss_per_step = []
    self.reward_per_step = []
    self.invalid_actions_per_round = 0
    self.escaped_bombs = 0
    

    self.logger.debug(f"Total Reward: {reward}")

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
