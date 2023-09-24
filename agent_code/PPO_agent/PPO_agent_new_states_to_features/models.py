import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import matplotlib.pyplot as plt
from .ReplayMemory import ReplayMemory
import random

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTIONS_NUM = [0, 1, 2, 3, 4, 5]

UP = [0,-1]
RIGHT = [1,0]
DOWN = [0,1]
LEFT = [-1,0]
WAIT = [0,0]

MOVES = {'UP': UP, 'RIGHT': RIGHT, 'DOWN': DOWN, 'LEFT': LEFT, 'WAIT': [0,0], 'BOMB': [0,0]}

class ActorModel(nn.Module):
    def __init__(self, n_actions, input_dims, alpha,
            Layer1_dims=5*5, Layer2_dims=5*10, save_dir='C:\\Users\\Maximilian\\Desktop\\bomberman_rl\\agent_code\\PPO_agent_new_state_to_feature'):
        super(ActorModel, self).__init__()

        self.save_file = os.path.join(save_dir, 'actor_torch_ppo')
        self.actor = nn.Sequential(
            nn.Linear(input_dims[0] * input_dims[1] * input_dims[2], Layer1_dims),
            nn.ReLU(),
            nn.Linear(Layer1_dims, Layer2_dims),
            nn.ReLU(),
            nn.Linear(Layer2_dims, n_actions),
            nn.Softmax(dim=-1))


        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        distribution = self.actor(state)
        distribution = Categorical(distribution)
        
        return distribution

    def save_save(self):
        T.save(self.state_dict(), self.save_file)

    def load_save(self):
        self.load_state_dict(T.load(self.save_file))

class CriticModel(nn.Module):
    def __init__(self, input_dims, alpha, Layer1_dims=5*5, Layer2_dims=5*10,
            save_dir='C:\\Users\\Maximilian\\Desktop\\bomberman_rl\\agent_code\\PPO_agent_new_state_to_feature'):
        super(CriticModel, self).__init__()

        self.save_file = os.path.join(save_dir, 'critic_torch_ppo')
        self.critic = nn.Sequential(
            nn.Linear(input_dims[0] * input_dims[1] * input_dims[2], Layer1_dims),
            nn.ReLU(),
            nn.Linear(Layer1_dims, Layer2_dims),
            nn.ReLU(),
            nn.Linear(Layer2_dims, 1))

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)

        return value

    def save_save(self):
        T.save(self.state_dict(), self.save_file)

    def load_save(self):
        self.load_state_dict(T.load(self.save_file))



class Agent:
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0003, gae_lambda=0.95, #0.95 und gamma = 0.99,
            policy_clip=0.2, batch_size=32, n_epochs = 5):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.gae_lambda = gae_lambda
        self.n_epochs = n_epochs
        self.step_scores = []
        self.game_scores = []
        self.game_iterations = 0
        self.min_score = -700
        self.avg_score = -10
        self.best_score = -1
        
        self.epsilon = 1
        self.epsilon_end = 0.05
        self.epsilon_decay = 0.999

        self.actor = ActorModel(n_actions, input_dims, alpha)
        self.critic = CriticModel(input_dims, alpha)
        self.memory = ReplayMemory(batch_size)
        
       
    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('saving model to save file')
        self.actor.save_save()
        self.critic.save_save()

    def load_models(self):
        print('loading model from save file')
        self.actor.load_save()
        self.critic.load_save()
    
    def clear_step_scores(self):
        self.step_scores = []
        
    def clear_game_scores(self):
        self.game_scores = []
        
    def store_step_scores(self, step_scores):
        self.step_scores.append(step_scores)
    
    def store_game_scores(self, game_scores):
        self.game_scores.append(game_scores)
        
    def set_min_score (self, score):
        self.min_score = score
        
    def give_back_action(self, game_state, train_state):
        state = state_to_features(game_state)
        features_tensor = T.from_numpy(state)
        features_tensor = features_tensor.flatten()

        distribution = self.actor(features_tensor)
        value = self.critic(features_tensor)

        # Epsilon-greedy exploration
        if train_state == True:
            rand = random.random()
            print("Epsilon:", self.epsilon)
            print("rand:", rand)
            if rand <= self.epsilon:
                print("random")
                action = random.randint(0, len(ACTIONS) - 1)
                action = T.tensor(action)
                if self.epsilon > self.epsilon_end:
                    self.epsilon *= self.epsilon_decay
                else:
                    self.epsilon = self.epsilon_end
            else:
                print("distribution")
                with T.no_grad():
                    #action = distribution.sample()
                    #action = T.argmax(distribution.sample()).item()
                    #action = T.tensor(action)
                    action = distribution.probs.argmax()
        else:
            with T.no_grad():
                #action = distribution.sample()
                #action = T.argmax(distribution.sample()).item()
                #action = T.tensor(action)
                action = distribution.probs.argmax()
        
        
        #print("11:", action)
        #print(distribution.sample())
        print("Probs:", distribution.probs)
        #action = T.tensor(action)
        #print("2:", action)
        
        #action = distribution.probs.argmax()
        action = ACTIONS[action.item()]
        print("action: ", action)

        return action
    
    def give_back_all (self, game_state, train_state):
        state = state_to_features(game_state)
        features_tensor = T.from_numpy(state)
        features_tensor = features_tensor.unsqueeze(0)
        features_tensor = features_tensor.flatten()
    
        distribution = self.actor(features_tensor)
        value = self.critic(features_tensor)

        value = T.squeeze(value).item()

        
        # Epsilon-greedy exploration
        if train_state == True:
            rand = random.random()               
            print("Epsilon:", self.epsilon)
            print("rand:", rand)
            if rand <= self.epsilon:
                print("random")
                action = random.randint(0, len(ACTIONS) - 1)
                action = T.tensor(action)
                if self.epsilon > self.epsilon_end:
                    self.epsilon *= self.epsilon_decay
                else:
                    self.epsilon = self.epsilon_end
            else:
                print("model")
                with T.no_grad():
                    #action = distribution.sample()
                    #action = T.argmax(distribution.sample()).item()
                    #action = T.tensor(action)
                    action = distribution.probs.argmax()
        else:
            with T.no_grad():
                #action = distribution.sample()
                #action = T.argmax(distribution.sample()).item()
                #action = T.tensor(action)
                action = distribution.probs.argmax()
        
        #action = distribution.sample()
        #action = np.argmax(distribution)
        #action = distribution.index(max(distribution))
        #print("11:", action)
        print("Probs:", distribution.probs)
        #action = T.tensor(action)
        #print("2:", action)
        
        
        probs = T.squeeze(distribution.log_prob(action)).item()
        action = ACTIONS[action.item()]
        #print(action.item())
        print("action: ", action)

        return action, probs, value, self.epsilon
    
def learn(self):
        for a in range(self.n_epochs):
            states, actions, previous_probability, values, rewards, finishs, batches = self.memory.generate_batches()

            accumulated_benefit = np.zeros(len(rewards), dtype=np.float32)

            for b in range(len(rewards)-1):
                accumulated_reduction = 1
                beginn = 0
                for c in range(b, len(rewards)-1):
                    beginn = beginn + (accumulated_reduction*(rewards[c] + self.gamma*values[c+1]*(1-int(finishs[c]))-values[c]))
                    accumulated_reduction = accumulated_reduction*(self.gae_lambda*self.gamma)
                accumulated_benefit[b] = beginn
            accumulated_benefit = T.tensor(accumulated_benefit).to(self.actor.device)
            
            
            values = T.tensor(values).to(self.actor.device)
            for batch in batches:
                states = states[batch]
                probability_old = T.tensor(previous_probability[batch]).to(self.actor.device)
                
                actions = actions[batch]
                # Convert action labels to numeric values using ACTIONS_NUM
                numeric_actions = [ACTIONS_NUM[ACTIONS.index(action)] for action in actions]
                # Convert the numeric actions to a PyTorch tensor
                actions_tensor = T.tensor(numeric_actions, dtype=T.int64).to(self.actor.device)  # Assuming self.actor.device is set

                #print(states)
                features_tensor = T.from_numpy(states)
                features_tensor = features_tensor.unsqueeze(0)
                
                #print(states.shape)
                distribution = self.actor(features_tensor)
                critic_value = self.critic(features_tensor)

                critic_value = T.squeeze(critic_value)

                new_probs = distribution.log_prob(actions_tensor)
                #print("Probs:", new_probs)
                prob_ratio = new_probs.exp()/(probability_old.exp())
                #prob_ratio = (new_probs - probability_old).exp()
                weighted_probs =  prob_ratio * accumulated_benefit[batch] 
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip) * accumulated_benefit[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = values[batch] + accumulated_benefit[batch]
                
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()
                total_loss = 0.5*critic_loss + actor_loss 
                
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()              
        return total_loss
    
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
            escape_moves = sorted(escape_moves, key=lambda x: x[1])
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
                for move in valid_moves:
                    if np.linalg.norm(np.array(move) + np.array(last_move)) == 0:
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

        print("SIZE:", np.array([go_up, go_right, go_down, go_left, drop_bomb]))
    
        return np.array([go_up, go_right, go_down, go_left, drop_bomb]).astype(np.float32)
    



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
        if np.min([np.linalg.norm(agent_position-other[3]) for other in others]) < 2:
            drop_bomb = 1

    return drop_bomb


def escape_bomb_recursively_bfs(self, agent_position, bomb_positions, game_state, possible_moves, depth=0):
    
    space = '' # this is just for logging --> remove later
    for i in range(depth):
        space += '  '

    if depth > 3:
        return depth, False
    
    for move in possible_moves:
        if not bombs_are_lethal(agent_position + move, bomb_positions):
            return depth, True
    
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

    

