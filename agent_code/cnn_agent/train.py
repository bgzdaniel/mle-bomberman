from collections import namedtuple, deque
import random

import torch
from torch import nn
from torch import optim
import numpy as np
import matplotlib.pyplot as plt

import pickle
from typing import List

import events as e
from .callbacks import state_to_features, DqnNet, get_bomb_rad_dict

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
TRANSITION_HISTORY_SIZE = int(1e9)
BATCH_SIZE = 32
DISCOUNT = 0.9

MOVE_ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT']

def sample(self, batch_size):
    return random.sample(self.transitions, batch_size)


def setup_training(self):
    self.target_net = DqnNet(self).to(self.device)
    self.target_net.load_state_dict(self.policy_net.state_dict())
    self.loss_function = nn.SmoothL1Loss()  # Huber Loss
    self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e3)
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.copy_weights_step = 200
    self.train_iter = 0
    self.scores = []
    self.round = 0

    with open("score_per_round.txt", "w") as file:
        file.write("round\tscore\n")

def reward_from_events(self, events: List[str]) -> int:
    game_rewards = {
        e.INVALID_ACTION: -5, # invalid actions waste time
        e.CRATE_DESTROYED: 1,
        e.COIN_FOUND: 2,
        e.COIN_COLLECTED: 10,
        e.KILLED_OPPONENT: 50,
        e.KILLED_SELF: -50,
        e.GOT_KILLED: -50,
        e.OPPONENT_ELIMINATED: 2.5,
        e.SURVIVED_ROUND: 1
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def evaluate_reward(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str], old_features, new_features):
    total_reward = 0

    # collect easy rewards from rules
    total_reward += reward_from_events(self, events)

    # get bomb coords and timers for whole radius and player coords
    new_bombs_rad = get_bomb_rad_dict(new_game_state)
    old_bombs_rad = get_bomb_rad_dict(old_game_state)
    new_player_coord = new_game_state["self"][3]
    old_player_coord = old_game_state["self"][3]

    # punish agent for being in bomb radius
    if new_player_coord in new_bombs_rad:
        total_reward += (new_bombs_rad[new_player_coord] - 5)
    # reward agent for stepping out of bomb radius
    elif old_player_coord in old_bombs_rad and new_player_coord not in new_bombs_rad:
        total_reward += ((old_bombs_rad[old_player_coord] - 5) * -1) * 0.75

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
        
        total_reward += (old_min_distance - new_min_distance) / 2.5
        print((old_min_distance - new_min_distance) / 2.5)

    return total_reward

    # TO-DO: reward agent for placing bombs which would hit other players and crates


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    old_features = state_to_features(self, old_game_state)
    new_features = state_to_features(self, new_game_state)

    reward = evaluate_reward(self, old_game_state, self_action, new_game_state, events, old_features, new_features)

    self.transitions.append(Transition(old_features, self.actions.index(self_action), new_features, reward))

    if self.train_iter % BATCH_SIZE:
        loss = update_params(self)

    if self.train_iter % BATCH_SIZE * 100:
        self.target_net.load_state_dict(self.policy_net.state_dict())

    self.train_iter += 1
    

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    score = last_game_state["self"][1]
    self.scores.append(score)

    with open("score_per_round.txt", "a") as file:
        file.write(f"{self.round}\t{score}\n")

    last_features = state_to_features(self, last_game_state)

    reward = reward_from_events(self, events)

    self.transitions.append(Transition(last_features, self.actions.index(last_action), None, reward))

    self.round += 1


def update_params(self):
    if len(self.transitions) < BATCH_SIZE:
        return

    replays = sample(self, BATCH_SIZE)
    replays.append(Transition(replays[1].state, replays[1].action, None, replays[1].reward))

    # calculate predictions
    replays_states = torch.cat([torch.from_numpy(replay.state)[None] for replay in replays]).to(self.device)
    replays_actions = torch.tensor([replay.action for replay in replays]).to(self.device)[:, None]
    predictions = torch.gather(self.policy_net(replays_states), 1, replays_actions)

    # calculate targets
    replays_non_terminal_states = torch.tensor([i for i, replay in enumerate(replays) if type(replay.next_state) is np.ndarray])
    replays_next_states = []
    for replay in replays:
        if type(replay.next_state) is not np.ndarray:
            replay_next_state = np.zeros(replays_states[0].shape, dtype=np.float32)
        else:
            replay_next_state = replay.next_state
        replays_next_states.append(torch.from_numpy(replay_next_state)[None])
    replays_next_states = torch.cat(replays_next_states).to(self.device)
    replays_rewards = torch.tensor([replay.reward for replay in replays]).to(self.device)[:, None]
    max_future_actions = torch.max(self.target_net(replays_next_states), dim=1)[0][:, None]
    targets = replays_rewards
    targets[replays_non_terminal_states, :] += DISCOUNT * max_future_actions[replays_non_terminal_states, :]

    # calculate loss, gradients and backpropagate
    loss = self.loss_function(predictions, targets)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    return loss