###### import numpy as np
import matplotlib.pyplot as plt
import os
import random
import json
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
from collections import namedtuple, deque
import pickle
import torch
from typing import List
from models_imitation import state_to_features
from models_imitation import Agent
import matplotlib.pyplot as plt
from collections import deque
from random import shuffle
import numpy as np
import settings as s
from collections import namedtuple, deque
import pickle
import torch
import keyboard
import numpy as np
from typing import List
import events as e
from .models import state_to_features
from .models import get_bomb_rad_dict
from .models import Agent
import settings
import matplotlib.pyplot as plt
from .utility import DataCollector

# Hyper parameters
BATCH_SIZE = 2 #32 works well
ALPHA = 0.0003 #or 0.0003
N_EPOCHS = 5 # 5 works well
N = 2 # 64 works well
N_GAMES = 10000
figure_file = 'C:\\Users\\Maximilian\\Desktop\\bomberman_rl\\agent_code\\PPO_agent\\scores.png'


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
TRANSITION_HISTORY_SIZE = int(1e6)
DISCOUNT = 0.9
saved_model_indexes = []


# Define the actions and their corresponding indices
MOVE_ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT']
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
n_actions = len(ACTIONS)
input_dims = (7, 17, 17)  # Define the input dimensions for the actor network


def setup_training(self):
    self.rounds = 0
    self.train_iter = 0
    self.reward_per_step = []
    self.loss_per_n_steps = []
    self.invalid_actions_per_round = 0
    #self.memory = ReplayMemory(BATCH_SIZE) nicht nötig alles ins replay memory from agent speichern
    self.data_collector = DataCollector("score_per_round.txt")
    self.data_collector.initialize()
    self.epsilon = 1
    self.train = True
    self.escaped_bombs = 0
    
    self.input_channels = 7

    self.conv_block_size = 1
    self.depth = 8
    self.init_channels = 32

    self.field_dim = 0
    self.bombs_dim = 1
    self.bombs_rad_dim = 2
    self.explosion_dim = 3
    self.coins_dim = 4
    self.myself_dim = 5
    self.other_dim = 6
    
    # Initialize the PPO agent
    #self.agent = Agent(n_actions = n_actions, batch_size = BATCH_SIZE, 
    #              alpha = ALPHA, input_dims = input_dims, n_epochs = N_EPOCHS)
    #agent = Agent(n_actions, input_dims)
    #self.agent.load_models()
    
    
    ################remove later
    self.before_action = "null"
    self.same_action_counter = 0
    self.action_list = []
    self.step_counter = 0
    self.no_bomb_counter = 0
     ##################

###wird jeden step aufgerufen
def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    
    #gives back the expert action, based on the rule_based agent
    expert_action = return_rule_based_action(old_game_state)
    expert_states_actions = np.concatenate((old_game_state.flatten(), expert_action))
    
        
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    reward = 0
    
    old_features = state_to_features(old_game_state)
    new_features = state_to_features(new_game_state)

    reward += reward_from_events(self, events)
    print("Action_Reward:", reward_from_actions(self, old_game_state, self_action, new_game_state, events, old_features, new_features))
    reward += reward_from_actions(self, old_game_state, self_action, new_game_state, events, old_features, new_features)
    print("Event_Reward:", reward_from_events(self, events))
    
    ############remove later########################
    self.action_list.append(self_action)
    self.step_counter += 1
    print("Step:", self.step_counter)
    
    #reward = -1*self.step_counter*0.4
    if self_action == self.before_action:
        self.same_action_counter += 1
        
    #if self_action == "UP":
    #    print("ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ")
    #    reward = reward - 8000
        
    #if self_action == "LEFT" or self_action == "RIGHT":
    #    reward = reward + 500
         
    #if self_action == "BOMB":
    #    reward = reward + 10000
    #else:
    #    reward = reward - 1000
    
    
    if self_action == "BOMB" and self.before_action == "BOMB":
        reward = reward - 20
    if self.before_action == "BOMB" and self_action == "UP":
        reward = reward + 10
    if self.before_action == "BOMB" and self_action == "DOWN":
        reward = reward + 10
    if self.before_action == "BOMB" and self_action == "LEFT":
        reward = reward + 10
    if self.before_action == "BOMB" and self_action == "RIGHT":
        reward = reward + 10
    if self_action == "BOMB" and self.step_counter == 1:
        reward = reward - 20
    if (self_action == "BOMB" and self.step_counter == 2) or (self_action == "BOMB" and self.step_counter == 3):
        reward = reward + 20

        
    if self.before_action == "UP" and self_action == "LEFT":
        reward = reward + 20
    if self.before_action == "UP" and self_action == "RIGHT":
        reward = reward + 20
    if self.before_action == "DOWN" and self_action == "LEFT":
        reward = reward + 20
    if self.before_action == "DOWN" and self_action == "RIGHT":
        reward = reward + 20
    if self.before_action == "LEFT" and self_action == "UP":
        reward = reward + 20
    if self.before_action == "RIGHT" and self_action == "UP":
        reward = reward + 20
    if self.before_action == "LEFT" and self_action == "DOWN":
        reward = reward + 20
    if self.before_action == "RIGHT" and self_action == "DOWN":
        reward = reward + 20
        
    #if self_action == self.before_action:
    #    reward = reward-0.3*self.step_counter
        
    if self_action != self.before_action:
        self.same_action_counter = 0
    if self.same_action_counter >= 8:
        print("8 in a row")
        reward = reward - 2*self.same_action_counter
    
    window_size = 4
    start_index = max(0, len(self.action_list) - 5)  # Start from the last 5 actions or the beginning if fewer than 5 actions
    for i in range(start_index, len(self.action_list) - window_size + 1):
        window = self.action_list[i:i + window_size]
        bomb_count = window.count("BOMB")

        if bomb_count > 2:
            reward -= 7.5
            self.no_bomb_counter = 0

        if bomb_count == 0:
            reward = reward - self.no_bomb_counter * 0.5
            self.no_bomb_counter += 1

        if bomb_count < 2 and bomb_count == 1:
            reward = reward + 25
            self.no_bomb_counter = 0
                
    window_size = 8
    # Get the most recent 8 actions from self.action_list
    recent_actions = self.action_list[-window_size:]
    print(recent_actions)
    # Find the index of the last "BOMB" in the recent_actions
    last_bomb_index = None
    for idx, action in enumerate(recent_actions[::-1]):
        if action == "BOMB":
            last_bomb_index = idx
            break
    # Check if there is a "BOMB" in recent_actions and if there are at least 5 actions after it
    if last_bomb_index is not None and last_bomb_index >= 4:
        print("BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB")
        reward += 250

        
    start_index = max(0, len(self.action_list) - 5)  # Start from the last 5 actions or the beginning if fewer than 5 actions
    window_size = 4
    if self.step_counter == 5:
        # Iterate through the list with a sliding window
        for i in range(start_index, len(self.action_list) - window_size + 1):
            bomb_count = self.action_list.count("BOMB")
            
            if bomb_count > 2:
                reward = reward - 5
    
            if bomb_count < 2 and bomb_count == 1:
                reward = reward + 50
                
            if bomb_count == 0:
                reward = reward - 5
                print("oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo")
                
    start_index = max(0, len(self.action_list) - 10)  # Start from the last 4 actions or the beginning if fewer than 5 actions
    window_size = 8
    for i in range(start_index, len(self.action_list) - window_size + 1):
        up_count = self.action_list.count("UP")
        down_count = self.action_list.count("DOWN")
        left_count = self.action_list.count("LEFT")
        right_count = self.action_list.count("RIGHT")
        bomb_count = self.action_list.count("BOMB")
            
        if up_count > 6:
            reward = reward - 75
        if down_count > 6:
            reward = reward - 75
        if left_count > 6:
            reward = reward - 75
        if right_count > 6:
            reward = reward - 75    
        if bomb_count > 3:
            reward = reward - 75
            
    self.before_action = self_action
  #######################################################################  
    
    
    print("Reward_end:", reward)
    self.reward_per_step.append(reward)

    if e.INVALID_ACTION in events:
        self.invalid_actions_per_round += 1

    
    
    ###alle events###
    value = 0
    probs = 0
    """while True:
        #print(old_game_state)
        action, prob, val, epsilon = self.agent.give_back_all(old_game_state, self.train)
        if action == self_action:
            break"""
    
    action, prob, val, epsilon = load_values_from_file('values.pkl')
    done = False
    score = reward
    self.epsilon = epsilon
    self.agent.remember((state_to_features(old_game_state)).flatten(), action, prob, val, reward, done)
    if len(self.agent.step_scores) % N == 0: #nach n steps wird der agents trainiert es wäre auch jeden step möglich allerdings ist es mit epochs stabiler
        total_loss = self.agent.learn_gail(expert_states_actions) #learning für den agent aus models_test.py
        print ("total loss:", total_loss)
        self.loss_per_n_steps.append(total_loss)
        plot_loss(self.loss_per_n_steps)
        self.agent.game_iterations += 1
    self.agent.store_step_scores(score)
    if keyboard.is_pressed("space"):
        self.agent.save_models()

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    
    save_values_to_file('values.pkl', 25, 0, 0, 0.5) #0.2
    
    ############remove later########################
    self.step_counter = 0
    self.before_action = "null"
    self.same_action_counter = 0
    self.no_bomb_counter = 0
    self.action_list = []      
    #############################################################
    
    
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    last_features = state_to_features(last_game_state)

    reward = reward_from_events(self, events) / 10
    

    self.reward_per_step.append(reward)

    if e.INVALID_ACTION in events:
        self.invalid_actions_per_round += 1


    self.train_iter += 1
    self.rounds += 1
    avg_invalid_actions_per_step = self.invalid_actions_per_round / last_game_state['step']
    killed_self = e.KILLED_SELF in events
    
    print("self loss:", self.loss_per_n_steps)

    self.data_collector.write(train_iter=self.train_iter, 
                              round=self.rounds, 
                              epsilon=self.epsilon, 
                              score=last_game_state["self"][1], 
                              killed_self=killed_self, 
                              avg_loss_per_step=np.mean([loss.detach().numpy() for loss in self.loss_per_n_steps]), 
                              avg_reward_per_step=np.mean(self.reward_per_step),
                              invalid_actions_per_round=self.invalid_actions_per_round,
                              avg_invalid_actions_per_step=avg_invalid_actions_per_step,
                              escaped_bombs=self.escaped_bombs)

    self.reward_per_step = []
    self.loss_per_n_steps = []
    self.invalid_actions_per_round = 0
    self.escaped_bombs = 0
    
    game_score = sum(self.agent.step_scores)/len(self.agent.step_scores)
    self.agent.store_game_scores(game_score)
    self.agent.clear_step_scores()
    
    if len(self.agent.game_scores) >= 10:
        self.agent.avg_score = np.mean(self.agent.game_scores[-10:])
        
    if self.agent.avg_score < -500:
        self.agent.load_models()
        
        
    print("min score:", self.agent.min_score)
    if self.agent.avg_score > self.agent.min_score:
        self.agent.best_score = self.agent.avg_score
        self.agent.set_min_score (self.agent.best_score)
        print("avg_score: ", self.agent.avg_score)
        print("min_score: ", self.agent.min_score)
        self.agent.save_models()
        if len(self.agent.game_scores)>5:
            saved_model_indexes.append(len(self.agent.game_scores) - 5)  # Add the index of the saved model
        else:
            saved_model_indexes.append(len(self.agent.game_scores) - 1)  # Add the index of the saved model
            
    x = [i+1 for i in range(len(self.agent.game_scores))]
    plot_learning_curve(x, self.agent.game_scores, figure_file, saved_model_indexes)
    self.logger.debug(f"Total Reward: {reward}")
    
    
# Define the reward_from_events function if not defined already


def reward_from_events(self, events: List[str]) -> int:
    total_reward = 0

    game_rewards = {
        e.INVALID_ACTION: -75,  #20# invalid actions waste time
        e.WAITED: -5,  # need for pro-active agent
        e.CRATE_DESTROYED: 50,
        e.COIN_FOUND: 100, #5,
        e.COIN_COLLECTED: 1000, #20,
        e.KILLED_OPPONENT: 10, #50,
        e.SURVIVED_ROUND: 100,  # note: the agent can only get this if you win the round or live until round 400
    }

    for event in events:
        if event in game_rewards:
            total_reward += game_rewards[event]

    if e.KILLED_SELF in events or e.GOT_KILLED in events:
        total_reward += -500 #-100

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

    bomb_scaling = 5
    # punish agent for being in bomb radius
    if new_player_coord in new_bombs_rad and e.BOMB_DROPPED not in events:
        total_reward += (new_bombs_rad[new_player_coord] - 4) * bomb_scaling
    # reward agent for stepping out of bomb radius
    elif old_player_coord in old_bombs_rad and new_player_coord not in new_bombs_rad:
        self.escaped_bombs += 1
        total_reward += (
            ((old_bombs_rad[old_player_coord] - 4) * bomb_scaling) * -1 * 100 #0.2
        )
        print("LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL:", ((old_bombs_rad[old_player_coord] - 4) * bomb_scaling) * -1 * 5 #
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
        diff = old_min_distance - new_min_distance
        coin_reward += diff * 2

        reward_for_coin_proximity = diff * 2
        # weight reward depending on distance to nearest coin
        reward_for_coin_proximity *= 1 / (new_min_distance) ** 2
        coin_reward += reward_for_coin_proximity

        self.logger.debug(f"Reward for coins: {coin_reward}")

        total_reward += coin_reward*100

    return total_reward
def plot_learning_curve(x, scores, figure_file, saved_model_indexes):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-10):(i+1)])
    
    plt.plot(x, running_avg)
    plt.title('Reward average of previous 10 games')
    
    for idx in saved_model_indexes:
        plt.annotate('Saved Model', (x[idx], running_avg[idx]), 
                     textcoords="offset points", xytext=(0,10), ha='center', fontsize=8,
                     arrowprops=dict(arrowstyle='->', color='r'))
    
    plt.savefig(figure_file)
    
    
    
def load_values_from_file(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    
    action = data['action']
    prob = data['prob']
    val = data['val']
    epsilon = data['epsilon']
    
    return action, prob, val, epsilon

def save_values_to_file(file_path, action, prob, val, epsilon):
    data = {
        'action': action,
        'prob': prob,
        'val': val,
        'epsilon': epsilon
    }
    
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)

def plot_loss(loss, filename='C:\\Users\\Maximilian\\Desktop\\bomberman_rl\\agent_code\\PPO_agent_new_state_to_feature\\PPO_loss_plot.png'):
    """
    Plots the average loss values over the last 'window_size' iterations.

    Parameters:
    loss (list): List of loss values to be plotted.
    filename (str): The filename to save the plot.
    window_size (int): The number of iterations to use for calculating the moving average.
    """
    
    iterations = list(range(1, len(loss) + 1))
    avg_loss = loss
    
    # Create a line plot of average loss vs. iterations
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, [l.detach().numpy() for l in avg_loss], marker='o', linestyle='-')
    plt.title('Average Loss of last steps')
    plt.xlabel('Iterations')
    plt.ylabel('Average Loss')
    plt.grid(True)
    plt.savefig(filename)
    

"""Use the code from rule based agent to give back the "expert_action"""

def look_for_targets(free_space, start, targets, logger=None):
    """Find direction of closest target that can be reached via free tiles.

    Performs a breadth-first search of the reachable free tiles until a target is encountered.
    If no target can be reached, the path that takes the agent closest to any target is chosen.

    Args:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.
        logger: optional logger object for debugging.
    Returns:
        coordinate of first step towards closest target or towards tile closest to any target.
    """
    if len(targets) == 0: return None

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)
        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            best = current
            break
        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]
        shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    if logger: logger.debug(f'Suitable target found at {best}')
    # Determine the first step towards the best found target tile
    current = best
    while True:
        if parent_dict[current] == start: return current
        current = parent_dict[current]

def return_rule_based_action(self, game_state):
    """
    Called each game step to determine the agent's next action.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.
    """
    self.logger.info('Picking action according to rule set')
    # Check if we are in a different round
    if game_state["round"] != current_round:
        current_round = game_state["round"]
    # Gather information about the game state
    arena = game_state['field']
    _, score, bombs_left, (x, y) = game_state['self']
    bombs = game_state['bombs']
    bomb_xys = [xy for (xy, t) in bombs]
    others = [xy for (n, s, b, xy) in game_state['others']]
    coins = game_state['coins']
    bomb_map = np.ones(arena.shape) * 5
    for (xb, yb), t in bombs:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i, j] = min(bomb_map[i, j], t)



    # Check which moves make sense at all
    directions = [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    valid_tiles, valid_actions = [], []
    for d in directions:
        if ((arena[d] == 0) and
                (game_state['explosion_map'][d] < 1) and
                (bomb_map[d] > 0) and
                (not d in others) and
                (not d in bomb_xys)):
            valid_tiles.append(d)
    if (x - 1, y) in valid_tiles: valid_actions.append('LEFT')
    if (x + 1, y) in valid_tiles: valid_actions.append('RIGHT')
    if (x, y - 1) in valid_tiles: valid_actions.append('UP')
    if (x, y + 1) in valid_tiles: valid_actions.append('DOWN')
    if (x, y) in valid_tiles: valid_actions.append('WAIT')


    # Collect basic action proposals in a queue
    # Later on, the last added action that is also valid will be chosen
    action_ideas = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    shuffle(action_ideas)

    # Compile a list of 'targets' the agent should head towards
    cols = range(1, arena.shape[0] - 1)
    rows = range(1, arena.shape[0] - 1)
    dead_ends = [(x, y) for x in cols for y in rows if (arena[x, y] == 0)
                 and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(0) == 1)]
    crates = [(x, y) for x in cols for y in rows if (arena[x, y] == 1)]
    targets = coins + dead_ends + crates
    # Add other agents as targets if in hunting mode or no crates/coins left

    # Exclude targets that are currently occupied by a bomb
    targets = [targets[i] for i in range(len(targets)) if targets[i] not in bomb_xys]

    # Take a step towards the most immediately interesting target
    free_space = arena == 0
    d = look_for_targets(free_space, (x, y), targets, self.logger)
    if d == (x, y - 1): action_ideas.append('UP')
    if d == (x, y + 1): action_ideas.append('DOWN')
    if d == (x - 1, y): action_ideas.append('LEFT')
    if d == (x + 1, y): action_ideas.append('RIGHT')
    if d is None:
        self.logger.debug('All targets gone, nothing to do anymore')
        action_ideas.append('WAIT')

    # Add proposal to drop a bomb if at dead end
    if (x, y) in dead_ends:
        action_ideas.append('BOMB')
    # Add proposal to drop a bomb if touching an opponent
    if len(others) > 0:
        if (min(abs(xy[0] - x) + abs(xy[1] - y) for xy in others)) <= 1:
            action_ideas.append('BOMB')
    # Add proposal to drop a bomb if arrived at target and touching crate
    if d == (x, y) and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(1) > 0):
        action_ideas.append('BOMB')

    # Add proposal to run away from any nearby bomb about to blow
    for (xb, yb), t in bombs:
        if (xb == x) and (abs(yb - y) < 4):
            # Run away
            if (yb > y): action_ideas.append('UP')
            if (yb < y): action_ideas.append('DOWN')
            # If possible, turn a corner
            action_ideas.append('LEFT')
            action_ideas.append('RIGHT')
        if (yb == y) and (abs(xb - x) < 4):
            # Run away
            if (xb > x): action_ideas.append('LEFT')
            if (xb < x): action_ideas.append('RIGHT')
            # If possible, turn a corner
            action_ideas.append('UP')
            action_ideas.append('DOWN')
    # Try random direction if directly on top of a bomb
    for (xb, yb), t in bombs:
        if xb == x and yb == y:
            action_ideas.extend(action_ideas[:4])

    # Pick last action added to the proposals list that is also valid
    while len(action_ideas) > 0:
        a = action_ideas.pop()

            return a


