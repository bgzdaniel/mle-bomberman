import random

import numpy as np
import torch
from torch import nn      

UP = [0,-1]
RIGHT = [1,0]
DOWN = [0,1]
LEFT = [-1,0]
WAIT = [0,0]

MOVES = {"UP": UP, "RIGHT": RIGHT, "DOWN": DOWN, "LEFT": LEFT, "WAIT": [0,0], "BOMB": [0,0]}
    
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

    self.action = None

    self.input_channels = 5

    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {self.device}\n')
    
    if self.train: # or not os.path.isfile("fc_agent_model.pth"): 
        self.policy_net = DqnNet(self).to(self.device)
    else:
        self.target_net = torch.load("fc_agent_model.pth", map_location=self.device)
        self.policy_net = torch.load("fc_agent_model.pth", map_location=self.device)


def act(self, game_state: dict) -> str:

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


    features = state_to_features(self, game_state)
    _features = _state_to_features(self,game_state)

    if not np.array_equal(features, _features):
        self.logger.debug('Diff:')
        self.logger.debug(f'X {int(_features[0])} X')
        self.logger.debug(f'{int(_features[3])} {int(_features[4])} {int(_features[1])}')
        self.logger.debug(f'X {int(_features[2])} X')

    # ----------------------------------- #
    # get index of non-zero features
    """
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
    """
    features = _features
    
    features = np.where(features != 0)[0]
    # randomly sample an index
    if len(features) == 0:
        return 'WAIT'
    index = np.random.choice(features)
    # get action from index
    action = self.actions[index]
    self.logger.debug(f'Action: {action}')
    return action
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
    self.action = self.actions[action]
    return self.actions[action]


def bomb_is_lethal(agent_position, bomb_position):
    if agent_position[0] != bomb_position[0] and agent_position[1] != bomb_position[1]:
        return False 
    if agent_position[0] == bomb_position[0] and np.abs(agent_position[1] - bomb_position[1]) <= 3:
        return True
    if agent_position[1] == bomb_position[1] and np.abs(agent_position[0] - bomb_position[0]) <= 3:
        return True
    return False

def state_to_features(self, game_state: dict):
    """
        Note: When refactoring chunks of the code, encapsulate it first in a function, add the refactored 
        code in a new function and verify correctness by using asserts and running several rounds
    """
    if game_state is None:
        return None

    agent_position = np.array(game_state['self'][3])

    go_north = 0
    go_south = 0
    go_west = 0
    go_east = 0
    drop_bomb = 0

    # where can the agent go?
    field = game_state['field']
    explosion_map = game_state['explosion_map']
    
    north = (agent_position[0], agent_position[1]-1)
    south = (agent_position[0], agent_position[1]+1)
    west = (agent_position[0]-1, agent_position[1])
    east = (agent_position[0]+1, agent_position[1])


    if field[north] == 0 and explosion_map[north] == 0:
        go_north = 1
    if field[south] == 0 and explosion_map[south] == 0:
        go_south = 1
    if field[west] == 0 and explosion_map[west] == 0:
        go_west = 1
    if field[east] == 0 and explosion_map[east] == 0:
        go_east = 1

    # should the agent drop a bomb?
    if game_state['self'][2]:
        agent_surroundings = np.array([field[north], field[south], field[west], field[east]])
        # drop bomb if 3 or more crates surround agent
        if np.count_nonzero(agent_surroundings == 1) >= 2:
            drop_bomb = 1
        # drop bomb if there's 1 crate and 2 walls around agent
        elif np.count_nonzero(agent_surroundings == 1) == 1 and np.count_nonzero(agent_surroundings == -1) == 2:
            drop_bomb = 1
        
        if len(game_state['others']) > 0:
          if np.min([np.linalg.norm(agent_position-other[3]) for other in game_state['others']]) < 3:
              self.logger.debug('dropped bomb due to enemy in range')
              drop_bomb = 1

    # if there's a bomb close to the agent, it should only see directions away from the bomb
    bombs = game_state["bombs"]
    if len(bombs) > 0:
        bomb_distances = np.array([np.linalg.norm(np.array(bomb[0]) - np.array(agent_position)) for bomb in bombs])
        closest_bomb_index = np.argmin(bomb_distances)
        closest_bomb = bombs[closest_bomb_index]
        if bomb_is_lethal(agent_position, closest_bomb[0]):
            drop_bomb = 0
            escape_north = 0
            escape_south = 0
            escape_west = 0
            escape_east = 0
            escape_found = False
            if go_north:
                if not bomb_is_lethal(agent_position+[0,-1], closest_bomb[0]):
                    escape_north = 1
                    escape_found = True
            if go_south:
                if not bomb_is_lethal(agent_position+[0,1], closest_bomb[0]) and not escape_found:
                    escape_south = 1
                    escape_found = True
            if go_west:
                if not bomb_is_lethal(agent_position+[-1,0], closest_bomb[0]) and not escape_found:
                    escape_west = 1
                    escape_found = True
            if go_east:
                if not bomb_is_lethal(agent_position+[1,0], closest_bomb[0]) and not escape_found:
                    escape_east = 1
                    escape_found = True

            if escape_found:
                self.logger.debug('Escape found')
                go_north = escape_north
                go_south = escape_south
                go_west = escape_west
                go_east = escape_east
            else:
                # only allow directions away from the bomb (when agent on bomb, just go anywhere valid)
                if bomb_distances[closest_bomb_index] > 0:
                    bomb_direction = np.array(agent_position) - np.array(closest_bomb[0])
                    if bomb_direction[1] > 0:
                        go_north = 0
                    if bomb_direction[1] < 0:
                        go_south = 0
                    if bomb_direction[0] > 0:
                        go_west = 0
                    if bomb_direction[0] < 0:
                        go_east = 0

        # now check if moving might get agent into bomb radius
        elif bomb_distances[closest_bomb_index] <= 5: # not sure if 5 is correct
            self.logger.debug('might get into bomb radius')
            if go_north and bomb_is_lethal(agent_position+[0,-1], closest_bomb[0]):
                go_north = 0
            if go_south and bomb_is_lethal(agent_position+[0,1], closest_bomb[0]):
                go_south = 0
            if go_west and bomb_is_lethal(agent_position+[-1,0], closest_bomb[0]):
                go_west = 0
            if go_east and bomb_is_lethal(agent_position+[1,0], closest_bomb[0]):
                go_east = 0
    
    # when the agent is not dodging bombs, he should not just move back and forth
    if self.action is not None and go_north + go_south + go_west + go_east > 1:
        match self.action:
            case 'UP':
                go_south = 0
            case 'DOWN':
                go_north = 0
            case 'LEFT':
                go_east = 0
            case 'RIGHT':
                go_west = 0
    
    if drop_bomb == 1:
      go_north = go_south = go_west = go_east = 0
    else:
        # guide agent towards coins
        coins = np.array(game_state["coins"])
        if len(coins) > 0 and go_north + go_south + go_west + go_east > 1:
            closet_coin_index = np.argmin(np.linalg.norm(coins - agent_position, axis=1))
            closest_coin_direction = coins[closet_coin_index] - agent_position

            array = []
             # 'UP', 'RIGHT', 'DOWN', 'LEFT', 'BOMB'
            if go_north != 0:
                array.append([[1,0,0,0,0,0], np.linalg.norm(closest_coin_direction-[0,-1])])
            if go_south != 0:
                array.append([[0,0,1,0,0,0], np.linalg.norm(closest_coin_direction-[0,1])])
            if go_west != 0:
                array.append([[0,0,0,1,0,0], np.linalg.norm(closest_coin_direction-[-1,0])])
            if go_east != 0:
                array.append([[0,1,0,0,0,0], np.linalg.norm(closest_coin_direction-[1,0])])

            self.logger.debug(f'array: {array}')
            # sort array by second value ascending
            array = sorted(array, key=lambda x: x[1])
            while len(array) > 1:
                array.pop()

            go_north = array[0][0][0]
            go_east = array[0][0][1]
            go_south = array[0][0][2]
            go_west = array[0][0][3]

    #TODO:
    # add handling for multiple bombs
    # add recursive algorithm to find escape route (maybe not necessary)
    
    self.logger.debug(f'X {go_north} X')
    self.logger.debug(f'{go_west} {drop_bomb} {go_east}')
    self.logger.debug(f'X {go_south} X')

    # 'UP', 'RIGHT', 'DOWN', 'LEFT', 'BOMB'
    #return np.array([go_north, go_east, go_south, go_west, drop_bomb]).flatten().astype(np.float32)
    return np.array([go_north, go_east, go_south, go_west, drop_bomb]).flatten().astype(np.float32)


def get_valid_moves(agent_position, game_state):
    field = game_state["field"]
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
    
def get_bomb_decision(agent_position, game_state):
    north = (agent_position[0], agent_position[1]-1)
    south = (agent_position[0], agent_position[1]+1)
    west = (agent_position[0]-1, agent_position[1])
    east = (agent_position[0]+1, agent_position[1])
    field = game_state["field"]
    drop_bomb = 0
    if game_state['self'][2]:
        agent_surroundings = np.array([field[north], field[south], field[west], field[east]])
        # drop bomb if 3 or more crates surround agent
        if np.count_nonzero(agent_surroundings == 1) >= 2:
            drop_bomb = 1
        # drop bomb if there's 1 crate and 2 walls around agent
        elif np.count_nonzero(agent_surroundings == 1) == 1 and np.count_nonzero(agent_surroundings == -1) == 2:
            drop_bomb = 1
        
        if len(game_state['others']) > 0:
          if np.min([np.linalg.norm(agent_position-other[3]) for other in game_state['others']]) < 3:
              drop_bomb = 1
    return drop_bomb

def escape_bomb_recursively(self, agent_position, bomb_positions, game_state, move, depth=0):

    if depth > 3:
        return depth, False
    
    space = ''
    for i in range(depth):
        space += '  '

    current_tile_save = bombs_are_lethal(agent_position, bomb_positions)
    
    #agent_position += move
    self.logger.debug(f'{space}Depth: {depth}')
    self.logger.debug(f'{space}Testing agent position: {agent_position+move}')
    if bombs_are_lethal(agent_position + move, bomb_positions):
        for move in get_valid_moves(agent_position, game_state):
            self.logger.debug(f'{space}Poss. Escape: {move}')
            escape_depth, escape_found = escape_bomb_recursively(self, agent_position + move, bomb_positions, game_state, move, depth+1)
            if escape_found:
                return escape_depth, True
        self.logger.debug(f'{space}No escape found')
        return depth, False
    self.logger.debug(f'{space}Escape found: {move}, depth: {depth}')
    return depth, True

    

def _state_to_features(self, game_state: dict):
    """
        Note: When refactoring chunks of the code, encapsulate it first in a function, add the refactored 
        code in a new function and verify correctness by using asserts and running several rounds
    """
    if game_state is None:
        return None

    agent_position = np.array(game_state['self'][3])

    valid_moves = get_valid_moves(agent_position, game_state)

    # should the agent drop a bomb?
    bomb_decision = get_bomb_decision(agent_position, game_state)

    bombs = game_state["bombs"]
    # since the recursive escape looks up to 3 steps into the future, we only consider bombs that are 6 steps away
    dangerous_bombs = np.array([bomb[0] for bomb in bombs if np.linalg.norm(np.array(bomb[0]) - np.array(agent_position))<=6])

    if len(dangerous_bombs) > 0:

        escape_moves = []
        for move in valid_moves + WAIT:
            depth, escape_found = escape_bomb_recursively(self, agent_position, dangerous_bombs, game_state, move)
            if escape_found:
                bomb_decision = 0
                escape_moves.append([move, depth])
                self.logger.debug(f'Final escape move: {move}')
        sorted(escape_moves, key=lambda x: x[1])
        self.logger.debug(escape_moves)
        if len(escape_moves) > 0:
            valid_moves = [escape_moves[0][0]]        
    

    if bomb_decision == 1:
      valid_moves = []
    else:
        # guide agent towards coins by removing moves that don't decrease distance to the closest coin
        coins = np.array(game_state["coins"])
        if len(coins) > 0 and len(valid_moves) > 1:
            closet_coin_index = np.argmin(np.linalg.norm(coins - agent_position, axis=1))
            closest_coin = coins[closet_coin_index]
            closest_coin_distance = np.linalg.norm(closest_coin - agent_position)

            moves_with_coin_distance = []
            for move in valid_moves:
                coin_distance = np.linalg.norm(closest_coin - (agent_position + move))
                moves_with_coin_distance.append([move, coin_distance])

            moves_with_coin_distance = sorted(moves_with_coin_distance, key=lambda x: x[1])
            self.logger.debug(moves_with_coin_distance)
            while len(moves_with_coin_distance) > 1:
                moves_with_coin_distance.pop()
            valid_moves = [move[0] for move in moves_with_coin_distance]

    # when the agent is not dodging bombs or chasing coins, he should not just move back and forth
    if self.action is not None and len(valid_moves) > 1:
        last_move = MOVES[self.action]
        for move in valid_moves:
            if move == last_move:
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
    
    return np.array([go_up, go_right, go_down, go_left, 0, drop_bomb]).flatten().astype(np.float32)
