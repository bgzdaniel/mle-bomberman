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

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))
TRANSITION_HISTORY_SIZE = int(1e6)
DISCOUNT = 0.99

MOVE_ACTIONS = [e.MOVED_UP, e.MOVED_RIGHT, e.MOVED_DOWN, e.MOVED_LEFT]


def sample(self, batch_size):
    weights = []
    for replay in self.transitions:
        weight = 2 if type(replay.next_state) is not np.ndarray else 1
        weights.append(weight)
    return random.choices(self.transitions, weights=weights, k=batch_size)


def setup_training(self):
    self.batch_size = 128
    self.target_net = DqnNet(self).to(self.device)
    self.target_net.load_state_dict(self.policy_net.state_dict())
    self.target_net.train()
    self.loss_function = nn.SmoothL1Loss()
    self.optimizer = optim.Adam(self.policy_net.parameters())
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.steps_per_copy = 2048
    self.train_iter = 0

    # for logging
    self.round = 1
    self.loss_per_step = []
    self.reward_per_step = []
    self.invalid_actions_per_round = 0
    self.weights_copied_iter = 0
    self.escaped_bombs = 0

    self.data_collector = DataCollector()
    self.data_collector.initialize()

    self.reward_scaling = 100


def reward_from_events(self, events: List[str]) -> int:
    total_reward = 0

    game_rewards = {
        e.INVALID_ACTION: -5,  # invalid actions waste time
        e.WAITED: -2,  # need for pro-active agent
        e.CRATE_DESTROYED: 5,
        e.COIN_FOUND: 5,
        e.COIN_COLLECTED: 10,
        e.KILLED_OPPONENT: 50,
        e.SURVIVED_ROUND: 100,  # note: the agent can only get this if you win the round or live until round 400
    }

    for event in events:
        if event in game_rewards:
            total_reward += game_rewards[event]

    if e.KILLED_SELF in events or e.GOT_KILLED in events:
        total_reward += -100

    self.logger.debug(f"Reward from events: {total_reward}")
    return total_reward


def distance_to_nearest_coin(self, feature_maps, game_state):
    field = feature_maps[self.field_dim]
    coin_coords = game_state["coins"]
    player_coord = game_state["self"][3]
    x, y = player_coord
    visited_coords = {player_coord: 0}
    coin_distances = []
    queue = deque([((x - 1, y), 1), ((x + 1, y), 1), ((x, y + 1), 1), ((x, y - 1), 1)])
    while True:
        if len(coin_distances) == len(coin_coords) or len(queue) == 0:
            break
        coord, distance = queue.popleft()
        if field[coord[0], coord[1]] == -1 or coord in visited_coords:
            continue
        visited_coords[coord] = distance
        if coord in coin_coords:
            coin_distances.append(distance)
        x, y = coord
        next_coords_distances = [
            ((x - 1, y), distance + 1),
            ((x + 1, y), distance + 1),
            ((x, y + 1), distance + 1),
            ((x, y - 1), distance + 1),
        ]
        for next_coord, next_distance in next_coords_distances:
            if next_coord not in visited_coords:
                queue.append((next_coord, next_distance))

    min_distance = min(coin_distances)
    return min_distance


def reward_from_actions(
    self,
    old_game_state: dict,
    self_action: str,
    new_game_state: dict,
    events: List[str],
    old_features,
    new_features,
):
    total_reward = 0

    # get bomb coords and timers for whole radius and player coords
    new_bombs_rad = get_bomb_rad_dict(new_game_state)
    old_bombs_rad = get_bomb_rad_dict(old_game_state)
    new_player_coord = new_game_state["self"][3]
    old_player_coord = old_game_state["self"][3]

    bomb_scaling = 10
    # punish agent for being in bomb radius
    if new_player_coord in new_bombs_rad and e.BOMB_DROPPED not in events:
        total_reward += (new_bombs_rad[new_player_coord] - 4) * bomb_scaling
    # reward agent for stepping out of bomb radius
    elif old_player_coord in old_bombs_rad and new_player_coord not in new_bombs_rad:
        self.escaped_bombs += 1
        total_reward += (
            ((old_bombs_rad[old_player_coord] - 4) * bomb_scaling) * -1 * 0.8
        )

    self.logger.debug(f"Reward for bombs: {total_reward}")

    # reward agent if it places bombs which would hit other players
    if e.BOMB_DROPPED in events:
        bomb_reward = 0
        bomb_location = {}
        bomb_location["bombs"] = [(new_player_coord, 3)]
        bomb_rad_dict = get_bomb_rad_dict(bomb_location)
        for other in new_game_state["others"]:
            other_coord = other[3]
            max_distance = 3
            distance = np.linalg.norm(
                np.array(other_coord) - np.array(new_player_coord)
            )
            if other_coord in bomb_rad_dict:
                bomb_reward += (
                    max_distance * bomb_scaling
                )  # for placing bomb near other player
                bomb_reward += (
                    (max_distance + 1) - distance
                ) * bomb_scaling  # for placing bomb close to other player
                total_reward += bomb_reward

    if (
        any(event in MOVE_ACTIONS for event in events)
        and (e.COIN_COLLECTED not in events)
        and len(new_game_state["coins"]) > 0
        and len(old_game_state["coins"]) > 0
    ):
        coin_reward = 0

        old_min_distance = distance_to_nearest_coin(self, old_features, old_game_state)
        new_min_distance = distance_to_nearest_coin(self, new_features, new_game_state)

        coin_reward += old_min_distance - new_min_distance

        reward_for_coin_proximity = old_min_distance - new_min_distance
        # weight reward depending on distance to nearest coin
        reward_for_coin_proximity *= 1 / (new_min_distance) ** 2
        coin_reward += reward_for_coin_proximity

        self.logger.debug(f"Reward for coins: {coin_reward}")

        total_reward += coin_reward

    return total_reward


def game_events_occurred(
    self,
    old_game_state: dict,
    self_action: str,
    new_game_state: dict,
    events: List[str],
):
    self.logger.debug(
        f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}'
    )

    old_features = state_to_features(self, old_game_state)
    new_features = state_to_features(self, new_game_state)

    reward = 0
    reward += reward_from_events(self, events)
    reward += reward_from_actions(
        self,
        old_game_state,
        self_action,
        new_game_state,
        events,
        old_features,
        new_features,
    )

    reward /= self.reward_scaling
    self.reward_per_step.append(reward)

    if e.INVALID_ACTION in events:
        self.invalid_actions_per_round += 1

    self.transitions.append(
        Transition(old_features, self.actions.index(self_action), new_features, reward)
    )

    loss = update_params(self)

    # copy weights to target net after n steps
    if self.train_iter % self.steps_per_copy == 0 and self.train_iter != 0:
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.train()

        self.weights_copied_iter += 1
        self.logger.debug(
            f"weights copied to target net! ({self.weights_copied_iter} times)\n"
        )

    self.train_iter += 1

    self.logger.debug(f"Total Reward: {reward}")


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    self.logger.debug(
        f'Encountered event(s) {", ".join(map(repr, events))} in final step'
    )

    last_features = state_to_features(self, last_game_state)

    reward = reward_from_events(self, events) / self.reward_scaling

    self.reward_per_step.append(reward)

    if e.INVALID_ACTION in events:
        self.invalid_actions_per_round += 1

    self.transitions.append(
        Transition(last_features, self.actions.index(last_action), None, reward)
    )

    loss = update_params(self)

    # copy weights to target net after n steps
    if self.train_iter % self.steps_per_copy == 0 and self.train_iter != 0:
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.train()

        self.weights_copied_iter += 1
        self.logger.debug(
            f"weights copied to target net! ({self.weights_copied_iter} times)\n"
        )

    self.train_iter += 1
    self.round += 1
    avg_invalid_actions_per_step = (
        self.invalid_actions_per_round / last_game_state["step"]
    )
    killed_self = e.KILLED_SELF in events or e.GOT_KILLED in events

    self.data_collector.write(
        train_iter=self.train_iter,
        round=self.round,
        epsilon=self.epsilon,
        score=last_game_state["self"][1],
        killed_self=killed_self,
        avg_loss_per_step=np.mean(self.loss_per_step),
        avg_reward_per_step=np.mean(self.reward_per_step),
        invalid_actions_per_round=self.invalid_actions_per_round,
        avg_invalid_actions_per_step=avg_invalid_actions_per_step,
        steps_survived=last_game_state["step"],
    )

    self.loss_per_step = []
    self.reward_per_step = []
    self.invalid_actions_per_round = 0
    self.escaped_bombs = 0

    self.logger.debug(f"Total Reward: {reward}")

    if self.round % 2000 == 0:
        torch.save(self.policy_net.state_dict(), "agent.pt")


def update_params(self):
    if len(self.transitions) < self.batch_size:
        return

    replays = sample(self, self.batch_size)

    # calculate predictions
    replays_states = torch.cat(
        [torch.from_numpy(replay.state)[None] for replay in replays]
    ).to(self.device)
    replays_actions = torch.tensor([replay.action for replay in replays]).to(
        self.device
    )[:, None]
    state_forward = self.policy_net(replays_states)
    predictions = torch.gather(state_forward, 1, replays_actions)
    predictions = predictions.type(torch.FloatTensor)

    # calculate targets
    replays_non_terminal_states = []
    for i, replay in enumerate(replays):
        if type(replay.next_state) is np.ndarray:
            replays_non_terminal_states.append(i)
    replays_non_terminal_states = torch.tensor(replays_non_terminal_states).to(
        self.device
    )

    replays_next_states = []
    for replay in replays:
        if type(replay.next_state) is np.ndarray:
            replays_next_states.append(torch.from_numpy(replay.next_state)[None])
    replays_next_states = torch.cat(replays_next_states).to(self.device)

    max_future_actions = torch.zeros(self.batch_size, 1).to(self.device)
    max_future_actions[replays_non_terminal_states, :] = torch.max(
        self.target_net(replays_next_states), dim=1
    )[0][:, None]

    replays_rewards = torch.tensor([replay.reward for replay in replays]).to(
        self.device
    )[:, None]
    targets = replays_rewards + DISCOUNT * max_future_actions
    targets = targets.type(torch.FloatTensor)

    # calculate loss, gradients and backpropagate
    loss = self.loss_function(predictions, targets)
    self.loss_per_step.append(loss.item())
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    return loss
