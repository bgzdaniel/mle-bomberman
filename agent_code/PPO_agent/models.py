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

class ActorModel(nn.Module):
    def __init__(self, n_actions, input_dims, alpha,
            Layer1_dims=17*20, Layer2_dims=119*20, save_dir='C:\\Users\\Maximilian\\Desktop\\bomberman_rl\\agent_code\\PPO_agent'):
        super(ActorModel, self).__init__()

        self.save_file = os.path.join(save_dir, 'PPO-actor')
        self.actor = nn.Sequential(
            nn.Linear(input_dims[0] * input_dims[1] * input_dims[2], Layer1_dims),
            nn.ReLU(),
            nn.Linear(Layer1_dims, Layer2_dims),
            nn.ReLU(),
            nn.Linear(Layer2_dims, n_actions),
            nn.Softmax(dim=-1))


        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.to(self.device)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

    def forward(self, state):
        distribution = self.actor(state)
        distribution = Categorical(distribution)
        
        return distribution

    def save_save(self):
        T.save(self.state_dict(), self.save_file)

    def load_save(self):
        self.load_state_dict(T.load(self.save_file))

class CriticModel(nn.Module):
    def __init__(self, input_dims, alpha, Layer1_dims=17*20, Layer2_dims=119*20,
            save_dir='C:\\Users\\Maximilian\\Desktop\\bomberman_rl\\agent_code\\PPO_agent'):
        super(CriticModel, self).__init__()

        self.save_file = os.path.join(save_dir, 'PPO-critic')
        self.critic = nn.Sequential(
            nn.Linear(input_dims[0] * input_dims[1] * input_dims[2], Layer1_dims),
            nn.ReLU(),
            nn.Linear(Layer1_dims, Layer2_dims),
            nn.ReLU(),
            nn.Linear(Layer2_dims, 1))

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.to(self.device)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

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

def get_bomb_rad_dict(game_state):
    bombs = {coords: timer for coords, timer in game_state["bombs"]}
    for coords, timer in game_state["bombs"]:
        for i in range(1, 3 + 1):
            x = coords[0]
            y = coords[1]
            bombradius = [(x, y - i), (x, y + i), (x - i, y), (x + i, y)]
            for bombrad_coord in bombradius:
                if bombrad_coord in bombs:
                    bombs[bombrad_coord] = min(timer, bombs[bombrad_coord])
                else:
                    bombs[bombrad_coord] = timer
    return bombs


def state_to_features(game_state: dict):
    if game_state is None:
        return None

    field = game_state["field"]
    #print("TYPE:", type(game_state["explosion_map"])) 
    field_shape = (17, 17)

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
            if (
                x >= 0
                and y >= 0
                and x < field_shape[0]
                and y < field_shape[1]
            ):
                if bombs_rad[x, y] != -1:
                    bombs_rad[x, y] = min(bombs_rad[x, y], timer)
                else:
                    bombs_rad[x, y] = timer

    explosion_map = game_state["explosion_map"]

    coins = np.zeros(field_shape, dtype=np.int32)
    if len(game_state["coins"]) != 0:
        coin_coords = np.array(game_state["coins"])
        coins[coin_coords[:, 0], coin_coords[:, 1]] = 1

    myself = np.zeros(field_shape, dtype=np.int32)
    myself_bomb_action = 1 if game_state["self"][2] else -1
    myself[game_state["self"][3][0], game_state["self"][3][1]] = myself_bomb_action

    others = np.zeros(field_shape, dtype=np.int32)
    if len(game_state["others"]) != 0:
        others_coords = np.array([coords for _, _, _, coords in game_state["others"]])
        others_bomb_action = np.array(
            [bomb_action for _, _, bomb_action, _ in game_state["others"]]
        )
        others_bomb_action = np.where(others_bomb_action == True, 1, -1)
        others[others_coords[:, 0], others_coords[:, 1]] = others_bomb_action

    channels = [field, bombs, bombs_rad, explosion_map, coins, myself, others]
    feature_maps = np.stack(channels, axis=0).astype(np.float32)
    return feature_maps