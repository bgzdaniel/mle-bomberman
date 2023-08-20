from collections import namedtuple, deque
import pickle
import torch
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
BATCH_SIZE = 5
ALPHA = 0.0003
N_EPOCHS = 10
N = 20
N_GAMES = 10000
figure_file = 'C:\\Users\\Maximilian\\Desktop\\bomberman_rl\\agent_code\\PPO_agent\\scores.png'


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
TRANSITION_HISTORY_SIZE = int(1e6)
DISCOUNT = 0.9


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
    self.epsilon = 0
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
    self.agent = Agent(n_actions = n_actions, batch_size = BATCH_SIZE, 
                  alpha = ALPHA, input_dims = input_dims, n_epochs = N_EPOCHS)
    #agent = Agent(n_actions, input_dims)
    self.agent.load_models()

###wird jeden step aufgerufen
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

    
    
    ###alle events###
    value = 0
    probs = 0
    while True:
        #print(old_game_state)
        print("self action:", self_action)
        action, prob, val, epsilon = self.agent.give_back_all(old_game_state, self.train)
        print ("action:", action)
        if action == self_action:
            break
    print("done")
    done = False
    score = reward
    self.epsilon = epsilon
    self.agent.remember((state_to_features(self, old_game_state)).flatten(), action, prob, val, reward, done)
    if len(self.agent.step_scores) % N == 0: #nach n steps wird der agents trainiert es wäre auch jeden step möglich allerdings ist es mit epochs stabiler
        total_loss = self.agent.learn() #learning für den agent aus models.py
        print ("total loss:", total_loss)
        self.loss_per_n_steps.append(total_loss)
        self.agent.game_iterations += 1
    self.agent.store_step_scores(score)

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    last_features = state_to_features(self, last_game_state)

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
    
    if len(self.agent.game_scores) >= 100:
        self.agent.avg_score = np.mean(self.agent.game_scores[-100:])

    if self.agent.avg_score > self.agent.min_score:
        self.agent.best_score = self.agent.avg_score
        self.agent.set_min_score (self.agent.best_score)
        print("avg_score: ", self.agent.avg_score)
        print("min_score: ", self.agent.min_score)
        self.agent.save_models()
            
    x = [i+1 for i in range(len(self.agent.game_scores))]
    plot_learning_curve(x, self.agent.game_scores, figure_file)
            
    self.logger.debug(f"Total Reward: {reward}")
    
    
# Define the reward_from_events function if not defined already
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


def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)
