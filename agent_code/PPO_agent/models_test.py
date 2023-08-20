import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import matplotlib.pyplot as plt
#from .ReplayMemory import ReplayMemory

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTIONS_NUM = [0, 1, 2, 3, 4, 5]

import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import matplotlib.pyplot as plt

class ReplayMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
        
        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        #print(n_states)
        #print(np.array(self.states))
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []


class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha,
            fc1_dims=17*20, fc2_dims=119*20, chkpt_dir='C:\\Users\\Maximilian\\Desktop\\bomberman_rl\\agent_code\\PPO_agent'):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')
        self.actor = nn.Sequential(
            nn.Linear(input_dims[0] * input_dims[1] * input_dims[2], fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim=-1)
        )


        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        dist = self.actor(state)
        dist = Categorical(dist)
        
        return dist

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=17*20, fc2_dims=119*20,
            chkpt_dir='C:\\Users\\Maximilian\\Desktop\\bomberman_rl\\agent_code\\PPO_agent'):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')
        self.critic = nn.Sequential(
            nn.Linear(input_dims[0] * input_dims[1] * input_dims[2], fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)

        return value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class Agent:
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
            policy_clip=0.2, batch_size=64, n_epochs = 10):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.gae_lambda = gae_lambda
        self.n_epochs = n_epochs
        self.step_scores = []
        self.game_scores = []
        self.game_iterations = 0
        self.min_score = -500
        self.avg_score = 0
        self.one_game_score = 0
        self.best_score = -1

        self.actor = ActorNetwork(n_actions, input_dims, alpha)
        self.critic = CriticNetwork(input_dims, alpha)
        self.memory = ReplayMemory(batch_size)
       
    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
    
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

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)

        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()

        probs = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()

        return action, probs, value
    
    def give_back_action (self, game_state):
        state = state_to_features(game_state)
        features_tensor = T.from_numpy(state)
        features_tensor = features_tensor.flatten()
    
        dist = self.actor(features_tensor)
        #print(dist)
        value = self.critic(features_tensor)
        action = dist.sample()
        #print(action)
        #action = T.squeeze(action).item()
        #print("actin before argmax:", action)
        action = ACTIONS[action.item()]
        #print("action:", action)
        
        return action
    
    
    def give_back_all (self, game_state):
        state = state_to_features(game_state)
        features_tensor = T.from_numpy(state)
        features_tensor = features_tensor.unsqueeze(0)
        features_tensor = features_tensor.flatten()
    
        dist = self.actor(features_tensor)
        value = self.critic(features_tensor)
        action = dist.sample()

        probs = T.squeeze(dist.log_prob(action)).item()
        value = T.squeeze(value).item()
        action = ACTIONS[action.item()]

        return action, probs, value

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr,\
            reward_arr, dones_arr, batches = \
                    self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*\
                            (1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t
            advantage = T.tensor(advantage).to(self.actor.device)
            
            
            values = T.tensor(values).to(self.actor.device)
            for batch in batches:
                states = state_arr[batch]
                old_probs = T.tensor(old_prob_arr[batch]).to(self.actor.device)
                
                actions = action_arr[batch]
                # Convert action labels to numeric values using ACTIONS_NUM
                numeric_actions = [ACTIONS_NUM[ACTIONS.index(action)] for action in actions]
                # Convert the numeric actions to a PyTorch tensor
                actions_tensor = T.tensor(numeric_actions, dtype=T.int64).to(self.actor.device)  # Assuming self.actor.device is set

                #print(states)
                features_tensor = T.from_numpy(states)
                features_tensor = features_tensor.unsqueeze(0)

                dist = self.actor(features_tensor)
                critic_value = self.critic(features_tensor)

                critic_value = T.squeeze(critic_value)

                new_probs = dist.log_prob(actions_tensor)
                prob_ratio = new_probs.exp() / old_probs.exp()
                #prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip,
                        1+self.policy_clip)*advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5*critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()               

def get_bomb_rad_dict(game_state: dict):
    bomb_rad_dict = {}
    field = game_state["field"]

    for bomb in game_state["bombs"]:
        coords, timer = bomb
        if timer == 0:
            continue  # Ignore bombs that are about to explode

        x, y = coords
        bomb_rad_dict[(x, y)] = timer

        # Calculate the tiles affected by the explosion
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                if abs(dx) + abs(dy) <= 3:  # Tiles within the explosion radius
                    new_x, new_y = x + dx, y + dy

                    # Check for obstacles and blocked tiles
                    if 0 <= new_x < len(field) and 0 <= new_y < len(field):
                        if field[new_x][new_y] != -1:  # Check for obstacle
                            blocked = False
                            for i in range(1, max(abs(dx), abs(dy)) + 1):
                                if field[x + i * np.sign(dx)][y + i * np.sign(dy)] == -1:
                                    blocked = True
                                    break
                            if not blocked:
                                bomb_rad_dict[(new_x, new_y)] = timer

    return bomb_rad_dict


def state_to_features(game_state):
    if game_state is None:
        return None
    
    field = game_state["field"]
    field_shape = (17, 17)
    #print (field)
    

    bombs = np.full(field_shape, -1, dtype=np.int32)
    bombs_rad = np.full(field_shape, -1, dtype=np.int32)
    if len(game_state["bombs"]) != 0:
        bomb_coords = np.array([coords for coords, _ in game_state["bombs"]])
        bomb_timers = np.array([timer for _, timer in game_state["bombs"]])
        bombs[bomb_coords[:, 0], bomb_coords[:, 1]] = bomb_timers
        bomb_rad_dict = get_bomb_rad_dict(game_state)
        for coords, timer in bomb_rad_dict.items():
            x = coords[0]
            y = coords[1]
            if x >= 0 and y >= 0:
                if bombs_rad[x, y] != -1:
                    bombs_rad[x, y] = min(bombs_rad[x, y], timer)                    
                else:
                    bombs_rad[x, y] = timer
    
    #print (bombs)
    #print (bombs_rad)

    explosion_map = np.array(game_state["explosion_map"], dtype=np.int32)
    #print (explosion_map)


    coins = np.zeros(field_shape, dtype=np.int32)
    if len(game_state["coins"]) != 0:
        coin_coords = np.array(game_state["coins"])
        coins[coin_coords[:, 0], coin_coords[:, 1]] = 1
    
    #print (coins)

    myself = np.zeros(field_shape, dtype=np.int32)
    myself_bomb_action = 1 if game_state["self"][2] == True else -1
    myself[game_state["self"][3][0], game_state["self"][3][1]] = myself_bomb_action

    #print (myself)

    
    others = np.zeros(field_shape, dtype=np.int32)
    if len(game_state["others"]) != 0:
        others_coords = np.array([coords for _, _, _, coords in game_state["others"]])
        others_bomb_action = np.array([bomb_action for _, _, bomb_action, _ in game_state["others"]])
        others_bomb_action = np.where(others_bomb_action == True, 1, -1)
        others[others_coords[:, 0], others_coords[:, 1]] = others_bomb_action

    #print (others)

    
    channels = [field, bombs, bombs_rad, explosion_map, coins, myself, others]
    feature_maps = np.stack(channels, axis=0, dtype=np.float32)
    return feature_maps
    