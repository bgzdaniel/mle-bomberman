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
    self.steps_per_copy = 3000
    self.train_iter = 0
    
    # for logging
    self.scores = []
    self.round = 0
    self.reward_per_round = 0
    self.invalid_actions_per_round = 0
    self.weights_copied_iter = 0

    with open("score_per_round.txt", "w") as file:
        file.write("training_iter\t round\t epsilon\t score\t killed_self\t avg_reward_per_step\t invalid_actions_per_round\n")

def reward_from_events(self, events: List[str]) -> int:
    total_reward = 0

    game_rewards = {
        e.INVALID_ACTION: -1, # invalid actions waste time
        e.WAITED: -0.5, # need for pro-active agent
        e.CRATE_DESTROYED: 2,
        e.COIN_FOUND: 3,
        e.COIN_COLLECTED: 15,
        e.KILLED_OPPONENT: 75,
        e.SURVIVED_ROUND: 75
    }

    for event in events:
        if event in game_rewards:
            total_reward += game_rewards[event]

    if e.KILLED_SELF or e.GOT_KILLED in events:
        total_reward += -75

    total_reward /= 10
    return total_reward


def evaluate_reward(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str], old_features, new_features):
    total_reward = 0

    # get bomb coords and timers for whole radius and player coords
    new_bombs_rad = get_bomb_rad_dict(new_game_state)
    old_bombs_rad = get_bomb_rad_dict(old_game_state)
    new_player_coord = new_game_state["self"][3]
    old_player_coord = old_game_state["self"][3]

    scaling = 3
    # punish agent for being in bomb radius
    if new_player_coord in new_bombs_rad:
        total_reward += ((new_bombs_rad[new_player_coord] - 4) * scaling)
    # reward agent for stepping out of bomb radius
    elif old_player_coord in old_bombs_rad and new_player_coord not in new_bombs_rad:
        total_reward += ((old_bombs_rad[old_player_coord] - 4) * scaling) * -1 * 0.25

    # reward agent for getting close to nearest coin
    if self_action in MOVE_ACTIONS:
        new_distances = []
        for coin_coord in new_game_state["coins"]:
            new_distances.append(np.linalg.norm(np.array(coin_coord) - np.array(new_player_coord)))
        new_min_distance = np.min(np.array(new_distances))

        old_distances = []
        for coin_coord in old_game_state["coins"]:
            old_distances.append(np.linalg.norm(np.array(coin_coord) - np.array(old_player_coord)))
        old_min_distance = np.min(np.array(old_distances))
        
        total_reward += (old_min_distance - new_min_distance) / 5

    total_reward /= 10
    return total_reward

    # TO-DO: reward agent for placing bombs which would hit other players and crates


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    old_features = state_to_features(self, old_game_state)
    new_features = state_to_features(self, new_game_state)

    reward = 0
    reward += 1 # agent survived
    reward += reward_from_events(self, events)
    reward += evaluate_reward(self, old_game_state, self_action, new_game_state, events, old_features, new_features)

    self.reward_per_round += reward

    if e.INVALID_ACTION in events:
        self.invalid_actions_per_round += 1

    self.transitions.append(Transition(old_features, self.actions.index(self_action), new_features, reward))
    
    loss = update_params(self)

    # copy weights to target net after n steps
    if self.train_iter % self.steps_per_copy == 0 and self.train_iter != 0:
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.weights_copied_iter += 1
        with open("score_per_round.txt", "a") as file:
            file.write(f"weights copied to target net! ({self.weights_copied_iter} times)\n")

    # increase batch size after every n steps for dampening of fluctuations
    # and faster convergence instead of decaying learning rate (https://arxiv.org/abs/1711.00489)
    if (self.train_iter % (self.steps_per_copy * 15) == 0) and (self.batch_size < 512) and self.train_iter != 0:
        self.batch_size *= 2
        with open("score_per_round.txt", "a") as file:
            file.write(f"batch size increased to {self.batch_size}!\n")

    self.train_iter += 1

    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    score = last_game_state["self"][1]
    self.scores.append(score)

    last_features = state_to_features(self, last_game_state)

    reward = reward_from_events(self, events)

    self.reward_per_round += reward

    if e.INVALID_ACTION in events:
        self.invalid_actions_per_round += 1

    self.transitions.append(Transition(last_features, self.actions.index(last_action), None, reward))

    loss = update_params(self)

    # copy weights to target net after n steps
    if self.train_iter % self.steps_per_copy == 0 and self.train_iter != 0:
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.weights_copied_iter += 1
        with open("score_per_round.txt", "a") as file:
            file.write(f"weights copied to target net! ({self.weights_copied_iter} times)\n")

    # increase batch size after every n steps for dampening of fluctuations
    # and faster convergence instead of decaying learning rate (https://arxiv.org/abs/1711.00489)
    if (self.train_iter % (self.steps_per_copy * 15) == 0) and (self.batch_size < 512) and self.train_iter != 0:
        self.batch_size *= 2
        with open("score_per_round.txt", "a") as file:
            file.write(f"batch size increased to {self.batch_size}!\n")

    self.train_iter += 1
    self.round += 1

    with open("score_per_round.txt", "a") as file:
        file.write(f"{self.train_iter}\t {self.round}\t {self.epsilon:.4f}\t {score}\t {e.KILLED_SELF in events}\t {(self.reward_per_round/last_game_state['step']):.4f}\t {self.invalid_actions_per_round}\n")
    self.reward_per_round = 0
    self.invalid_actions_per_round = 0

    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

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
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    return loss