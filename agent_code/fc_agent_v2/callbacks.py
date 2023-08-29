import random

import numpy as np
import torch
from torch import nn      

UP = [0,-1]
RIGHT = [1,0]
DOWN = [0,1]
LEFT = [-1,0]
WAIT = [0,0]

MOVES = {'UP': UP, 'RIGHT': RIGHT, 'DOWN': DOWN, 'LEFT': LEFT, 'WAIT': [0,0], 'BOMB': [0,0]}
    
class DqnNet(nn.Module):
    
    def __init__(self, outer_self):
        super(DqnNet, self).__init__()
        
        hidden_size = 8
        self.fc1 = nn.Linear(5, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def setup(self):
    self.actions = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

    self.epsilon = 1
    self.epsilon_end = 0.05
    self.epsilon_decay = 0.9975

    self.last_action = {'move': None, 'step': 1}

    self.input_channels = 5

    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {self.device}\n')
    
    if self.train: # or not os.path.isfile("fc_agent_model.pth"): 
        self.policy_net = DqnNet(self).to(self.device)
    else:
        self.target_net = torch.load('fc_agent_model.pth', map_location=self.device)
        self.policy_net = torch.load('fc_agent_model.pth', map_location=self.device)

def print_map_to_log(self, game_state):
    field = np.array(game_state['field'], dtype=object)
    for bomb in game_state['bombs']:
        field[bomb[0]] = '*'

    for coin in game_state['coins']:
        field[coin] = 'C'

    for opponent in game_state['others']:
        field[opponent[3]] = 'E'

    field[game_state['self'][3]] = 'A'  

    # replace all zeros in field with ' '
    field =  np.where(field == -1, '%', field)
    field =  np.where(field == 1, 'X', field)
    field = np.where(field == 0, ' ', field)
    field[(0,0)] = 'A'
    field[(0,16)] = 'B'
    field[(16,0)] = 'C'
    field[(16,16)] = 'D'
    self.logger.debug(f'\n{field.T}')


def act(self, game_state: dict) -> str:

    self.logger.debug('Map in act()')
    print_map_to_log(self, game_state)

    self.logger.debug(f'self.action = {self.last_action}')

    features = state_to_features(self, game_state)

    self.logger.debug(f'X {int(features[0])} X')
    self.logger.debug(f'{int(features[3])} {int(features[4])} {int(features[1])}')
    self.logger.debug(f'X {int(features[2])} X')

    self.logger.debug(features)

    """
    if not np.array_equal(features, _features):
        self.logger.debug('Diff:')

    # ----------------------------------- #
    # get index of non-zero features
    field = np.array(game_state['field'], dtype=object)
    for bomb in game_state['bombs']:
        field[bomb[0]] = '*'

    for coin in game_state['coins']:
        field[coin] = 'C'

    for opponent in game_state['others']:
        field[opponent[3]] = 'E'

    field[game_state['self'][3]] = 'A'  

    # replace all zeros in field with ' '
    field =  np.where(field == -1, '%', field)
    field =  np.where(field == 1, 'X', field)
    field = np.where(field == 0, ' ', field)
    self.logger.debug(f'\n{field.T}')
    
    features = np.where(features != 0)[0]
    # randomly sample an index
    if len(features) == 0:
        return 'WAIT'
    index = np.random.choice(features)
    # get action from index
    action = self.actions[index]
    self.logger.debug(f'Action: {action}')
    return action
    """
    # ----------------------------------- #

    if self.train == True:
        rand = random.random()
        if rand <= self.epsilon:
            action = random.randint(0, len(self.actions)-1)
            if self.epsilon > self.epsilon_end:
                self.epsilon *= self.epsilon_decay
            else:
                self.epsilon = self.epsilon_end
        else:
            with torch.no_grad():
                features = torch.from_numpy(features).to(self.device)[None]
                predictions = self.policy_net(features)
            action = torch.argmax(predictions).item()
    else:
        with torch.no_grad():
            features = torch.from_numpy(features).to(self.device)[None]
            predictions = self.policy_net(features)
            action = torch.argmax(predictions).item()
            self.logger.debug(f'Action: {self.actions[action]}')
    self.last_action = {'move': self.actions[action], 'step': game_state['step']}
    return self.actions[action]


def bomb_is_lethal(agent_position, bomb_position):
    if agent_position[0] != bomb_position[0] and agent_position[1] != bomb_position[1]:
        return False 
    if agent_position[0] == bomb_position[0] and np.abs(agent_position[1] - bomb_position[1]) <= 3:
        return True
    if agent_position[1] == bomb_position[1] and np.abs(agent_position[0] - bomb_position[0]) <= 3:
        return True
    return False

def get_valid_moves(agent_position, game_state):
    field = game_state["field"].copy()
    # treat opponents as walls to compute valid moves
    for other in game_state["others"]:
        field[other[3]] = -1
    explosion_map = game_state["explosion_map"]
    valid_moves = []
    for move in [UP, RIGHT, DOWN, LEFT]:
        new_position = tuple(agent_position + move)
        if field[new_position] == 0 and explosion_map[new_position] == 0:
            valid_moves.append(move)
    return valid_moves

def bombs_are_lethal(agent_position, bomb_positions):
    for bomb_position in bomb_positions:
        if bomb_is_lethal(agent_position, bomb_position):
            return True
    return False

def get_agent_surroudings(agent_position, game_state):
    up = tuple(agent_position + UP)
    down = tuple(agent_position + DOWN)
    left = tuple(agent_position + LEFT)
    right = tuple(agent_position + RIGHT)
    field = game_state["field"]
    agent_surroundings = np.array([field[up], field[down], field[left], field[right]])
    return agent_surroundings

def get_bomb_decision(agent_position, agent_surroundings, others):

    drop_bomb = 0
    
    # drop bomb if 3 or more crates surround agent
    if np.count_nonzero(agent_surroundings == 1) >= 2:
        drop_bomb = 1
    # drop bomb if there's 1 crate and 2 walls around agent
    elif np.count_nonzero(agent_surroundings == 1) == 1 and np.count_nonzero(agent_surroundings == -1) == 2:
        drop_bomb = 1
    
    if len(others) > 0:
        # idea: experiment with this parameter
        if np.min([np.linalg.norm(agent_position-other[3]) for other in others]) < 3:
            drop_bomb = 1

    return drop_bomb


def escape_bomb_recursively_bfs(self, agent_position, bomb_positions, game_state, possible_moves, depth=0):
    
    space = '' # this is just for logging --> remove later
    for i in range(depth):
        space += '  '

    if depth > 3:
        self.logger.debug(f'{space}No escape found: depth {depth}')
        return depth, False
    
    for move in possible_moves:
        if not bombs_are_lethal(agent_position + move, bomb_positions):
            self.logger.debug(f'{space}Escape found: {move} at depth {depth}')
            return depth, True
    
    self.logger.debug(f'Tested: {possible_moves} at depth {depth}')
    self.logger.debug(f'{space}No escape found at depth {depth} --> going one level deeper')
    for move in possible_moves:
        new_agent_position = agent_position + move
        new_possible_moves = get_valid_moves(new_agent_position, game_state)
        escape_depth, escape_found = escape_bomb_recursively_bfs(self, new_agent_position, bomb_positions, game_state, new_possible_moves, depth+1)
        if escape_found:
            return escape_depth, True
    return depth, False

def make_agent_go_closer_to(target_position, agent_position, valid_moves):
    moves_with_distance_to_target = []
    for move in valid_moves:
        target_distance = np.linalg.norm(target_position - (agent_position + move))
        moves_with_distance_to_target.append([move, target_distance])

    moves_with_distance_to_target = sorted(moves_with_distance_to_target, key=lambda x: x[1])
    while len(moves_with_distance_to_target) > 1:
        moves_with_distance_to_target.pop()
    return [move[0] for move in moves_with_distance_to_target]

    

def state_to_features(self, game_state: dict):
    """
        Note: When refactoring chunks of the code, encapsulate it first in a function, add the refactored 
        code in a new function and verify correctness by using asserts and running several rounds
    """
    if game_state is None:
        return None

    agent_position = np.array(game_state['self'][3])
    agent_surroundings = get_agent_surroudings(agent_position, game_state)
    valid_moves = get_valid_moves(agent_position, game_state)

    # should the agent drop a bomb?
    if not game_state['self'][2]:
        bomb_decision = 0
    else:
        bomb_decision = get_bomb_decision(agent_position, agent_surroundings, game_state['others'])

    bombs = game_state["bombs"]
    # since the recursive escape looks up to 3 steps into the future, we only consider bombs that are 6 steps away
    dangerous_bombs = np.array([bomb[0] for bomb in bombs if np.linalg.norm(np.array(bomb[0]) - np.array(agent_position))<=6])

    if len(dangerous_bombs) > 0:

        escape_moves = []
        for move in valid_moves + [WAIT]:
            depth, escape_found = escape_bomb_recursively_bfs(self, agent_position, dangerous_bombs, game_state, [move])
            if escape_found:
                bomb_decision = 0
                escape_moves.append([move, depth])
                self.logger.debug(f'Final escape move: {move}')
        escape_moves = sorted(escape_moves, key=lambda x: x[1])
        self.logger.debug(f'Possible Escapes: {escape_moves}')
        if len(escape_moves) > 0:
            valid_moves = [escape_moves[0][0]]        
    
    # when the agent chases a coin, disable bombs until it reaches a dead end
    coins = np.array(game_state['coins'])
    if bomb_decision == 1 and (len(coins) == 0 or np.count_nonzero(agent_surroundings != 0) >= 3):
      valid_moves = []
    else:
        # guide agent towards coins by removing moves that don't decrease distance to the closest coin
        if len(coins) > 0 and len(valid_moves) > 1:
            closet_coin_index = np.argmin(np.linalg.norm(coins - agent_position, axis=1))
            closest_coin = coins[closet_coin_index]

            valid_moves = make_agent_go_closer_to(closest_coin, agent_position, valid_moves)

        # and make agent go towards closest opponent
        if len(game_state['others']) > 1 and len(valid_moves):
            closest_opponent_index = np.argmin([np.linalg.norm(np.array(other[3]) - agent_position) for other in game_state['others']])
            closest_opponent = game_state['others'][closest_opponent_index]
            valid_moves = make_agent_go_closer_to(closest_opponent[3], agent_position, valid_moves)

        # when the agent is not dodging bombs or chasing coins, this prevents it from just moving back and forth
        if self.last_action['step'] < game_state['step'] and len(valid_moves) > 1:
            last_move = MOVES[self.last_action['move']]
            self.logger.debug(f'last_move: {last_move}')
            for move in valid_moves:
                if np.linalg.norm(np.array(move) + np.array(last_move)) == 0:
                    self.logger.debug(f'Removed move: {move}')
                    valid_moves.remove(move)
                    break
            

    go_up = go_right = go_down = go_left = 0
    drop_bomb = bomb_decision

    for move in valid_moves:
        if move == UP:
            go_up = 1
        if move == RIGHT:
            go_right = 1
        if move == DOWN:
            go_down = 1
        if move == LEFT:
            go_left = 1

    """
    self.logger.debug(f'X {go_up} X')
    self.logger.debug(f'{go_left} {drop_bomb} {go_right}')
    self.logger.debug(f'X {go_down} X')
    """
    
    return np.array([go_up, go_right, go_down, go_left, drop_bomb]).flatten().astype(np.float32)
