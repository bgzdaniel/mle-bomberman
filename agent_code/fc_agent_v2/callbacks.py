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
    """
    This class defines the neural net used by the agent.
    """
    
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
    """
    Setup function that is called once to initialize the agent. If requested, a new
    agent model is trained.
    """
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


def print_map_to_log(self, game_state) -> None:
    """
    Helper function that prints the current game state to log.

    :param game_state: Current state of the game
    """
    field = np.array(game_state['field'], dtype=object)
    for bomb in game_state['bombs']:
        field[bomb[0]] = '*'

    for coin in game_state['coins']:
        field[coin] = 'C'

    for opponent in game_state['others']:
        field[opponent[3]] = 'E'

    field[game_state['self'][3]] = 'A'  

    field =  np.where(field == -1, '%', field)
    field =  np.where(field == 1, 'X', field)
    field = np.where(field == 0, ' ', field)
    field[(0,0)] = 'A'
    field[(0,16)] = 'B'
    field[(16,0)] = 'C'
    field[(16,16)] = 'D'
    self.logger.debug(f'\n{field.T}')


def act(self, game_state: dict) -> str:
    """
    Get action from agent. The are two scenarios: Either we train or we play.
    The difference is epsilon decay and random choices during training.

    :param game_state: Current state of the game
    """

    self.logger.debug('Map in act()')
    print_map_to_log(self, game_state)

    self.logger.debug(f'self.action = {self.last_action}')

    features = state_to_features(self, game_state)

    self.logger.debug(f'X {int(features[0])} X')
    self.logger.debug(f'{int(features[3])} {int(features[4])} {int(features[1])}')
    self.logger.debug(f'X {int(features[2])} X')

    self.logger.debug(features)

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
    """
    This function determines whether a bomb close to the agent is deadly or not. Note how this function
    does not take the bomb timer into account.

    :param agent_position: Current agent position on the field
    :param bomb_position: Position of a bomb on the field
    :return: A boolean that indicates if the given bomb is deadly
    """
    if agent_position[0] != bomb_position[0] and agent_position[1] != bomb_position[1]:
        return False 
    if agent_position[0] == bomb_position[0] and np.abs(agent_position[1] - bomb_position[1]) <= 3:
        return True
    if agent_position[1] == bomb_position[1] and np.abs(agent_position[0] - bomb_position[0]) <= 3:
        return True
    return False

def get_valid_moves(agent_position, game_state):
    """
    This function returns the valid moves given a agent position and game state. Moves to a tile
    that contains a wall, crate, opponent or explosion are considered invalid.

    :param agent_position: Current agent position on the field
    :param game_state: Current state of the game
    """
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
    """
    This function computes the deadlines of several bombs given the agent's position.

    :param agent_position: Current agent position on the field
    :param bomb_positions: Array of bomb positions
    :return: A boolean that indicates if one of the bombs is deadly
    """
    for bomb_position in bomb_positions:
        if bomb_is_lethal(agent_position, bomb_position):
            return True
    return False

def get_agent_surroudings(agent_position, game_state):
    """
    This function returns the contents of the four fields that surround the agent.

    :param agent_position: Current agent position on the field
    :param game_state: Current state of the game
    :return: Array of length 4 that contains the agents surroundings
    """
    up = tuple(agent_position + UP)
    down = tuple(agent_position + DOWN)
    left = tuple(agent_position + LEFT)
    right = tuple(agent_position + RIGHT)
    field = game_state["field"]
    agent_surroundings = np.array([field[up], field[down], field[left], field[right]])
    return agent_surroundings

def get_bomb_decision(agent_position, agent_surroundings, others):
    """
    This function calculates whether the agent should drop a bomb given the agent position,
    the agent's surroundings and the positions of the other agents.

    :param agent_position: Current agent position on the field
    :param agent_surroundings: Array that contains the content of the 4 fields around the agent
    :param others: Array that contains the positions of the other agents
    :return: Decision to drop a bomb as boolean
    """
    drop_bomb = 0
    
    # drop bomb if 3 or more crates surround agent
    if np.count_nonzero(agent_surroundings == 1) >= 2:
        drop_bomb = 1
    # drop bomb if there's 1 crate and 2 walls around agent
    elif np.count_nonzero(agent_surroundings == 1) == 1 and np.count_nonzero(agent_surroundings == -1) == 2:
        drop_bomb = 1
    
    if len(others) > 0:
        # idea: experiment with this parameter
        if np.min([np.linalg.norm(agent_position-other[3]) for other in others]) < 2:
            drop_bomb = 1

    return drop_bomb


def escape_bomb_recursively_bfs(self, agent_position, bomb_positions, game_state, possible_moves, depth=0):
    """
    This function recursively finds the shortest escape path given the game state, agent position,
    bomb position and possible moves.

    :param agent_position: Current agent position on the field
    :param bomb_position: Array of bomb positions
    :param game_state: Current state of the game
    :param possible_moves: Array that contains all possible moves
    :return: 
        - The depth at which an escape was found
        - A boolean indicating if an escape was found
    """
    
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
    """
    This function invalides moves that guide the agent further away from a specified target.

    :param target_position: Position of the target that the agent should get closer to
    :param agent_position: Current agent position on the field
    :param valid_moves: Array of valid moves
    :return: The move that gets the agent closer to the given target
    """
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
    This function computes the features from the game state and is a core element of this agent.

    :param game_state: Current state of the game
    :return: Features as an array
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
            self.logger.debug(f'Valid moves: {valid_moves} (pre-coin)')
            valid_moves = make_agent_go_closer_to(closest_coin, agent_position, valid_moves)
            self.logger.debug(f'Valid moves: {valid_moves} (post-coin)')

        # and make agent go towards closest opponent
        if len(game_state['others']) > 1 and len(valid_moves):
            closest_opponent_index = np.argmin([np.linalg.norm(np.array(other[3]) - agent_position) for other in game_state['others']])
            closest_opponent = game_state['others'][closest_opponent_index]
            self.logger.debug(f'Valid moves: {valid_moves} (pre-oppo)')
            valid_moves = make_agent_go_closer_to(closest_opponent[3], agent_position, valid_moves)
            self.logger.debug(f'Valid moves: {valid_moves} (post-oppo)')

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
    
    return np.array([go_up, go_right, go_down, go_left, drop_bomb]).flatten().astype(np.float32)
