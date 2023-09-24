import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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
        
        self.probability = []
        self.finish = []
        self.value = []
        self.actions = []
        self.rewards = []
        self.states = []
    
        
        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        
        if n_states < self.batch_size:
            return np.array(self.states), np.array(self.actions), np.array(self.probability), np.array(self.value), np.array(self.rewards), np.array(self.finish), [np.arange(n_states)]
        #print(n_states)
        #print(np.array(self.states))
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states), np.array(self.actions), np.array(self.probability), np.array(self.values), np.array(self.rewards), np.array(self.finish), batches

    def store_memory(self, state, action, probability, values, reward, finish):
        self.states.append(state)
        self.actions.append(action)
        self.probability.append(probability)
        self.values.append(values)
        self.rewards.append(reward)
        self.finish.append(finish)

    def clear_memory(self):
        self.states = []
        self.values = []
        self.finish = []
        self.probability = []
        self.actions = []
        self.rewards = []

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
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)

        return value

    def save_save(self):
        T.save(self.state_dict(), self.save_file)

    def load_save(self):
        self.load_state_dict(T.load(self.save_file))

        
class Discriminator(nn.Module):
    def __init__(self, n_actions, input_dims, alpha, fc1_dims=512, fc2_dims=512,
                 chkpt_dir='C:\\Users\\Maximilian\\Desktop\\bomberman_rl\\agent_code\\PPO_agent'):
        super(Discriminator, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'discriminator')
        self.discriminator = nn.Sequential(
            nn.Linear(n_actions + (input_dims[0] * input_dims[1] * input_dims[2]), fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1),
            nn.Sigmoid()
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state_actions):
        prob = self.discriminator(state_actions)
        return prob

    def save_save(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_save(self):
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
        self.min_score = -5
        self.avg_score = 0
        self.one_game_score = 0
        self.best_score = -1

        self.actor = Actor(n_actions, input_dims, alpha)
        self.critic = Critic(input_dims, alpha)
        self.discriminator = Discriminator(n_actions, input_dims, alpha)
        self.memory = ReplayMemory(batch_size)
       
    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('saving model to save file')
        self.actor.save_save()
        self.critic.save_save()
        self.discriminator.save_save()

    def load_models(self):
        print('loading model from save file')
        self.actor.load_save()
        self.critic.load_save()
        self.discriminator.load_save()
    
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
 
    """def give_back_action (self, game_state):
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
        
        return action"""
    
    
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

        
    def learn_discriminator(self, expert_states_actions, agent_states_actions):
        self.discriminator.optimizer.zero_grad()
        
        expert_states_actions = T.tensor(expert_states_actions, dtype=T.float).to(self.device)
        agent_states_actions = T.tensor(agent_states_actions, dtype=T.float).to(self.device)
        
        expert_probs = self.discriminator(expert_states_actions)
        agent_probs = self.discriminator(agent_states_actions)
        
        expert_labels = T.ones(expert_probs.shape).to(self.device)
        agent_labels = T.zeros(agent_probs.shape).to(self.device)
        
        expert_loss = F.binary_cross_entropy(expert_probs, expert_labels)
        agent_loss = F.binary_cross_entropy(agent_probs, agent_labels)
        
        total_loss = expert_loss + agent_loss
        total_loss.backward()
        self.discriminator.optimizer.step()
        
        loss_values = [expert_loss.item(), agent_loss.item()]
        labels = ['Expert Loss', 'Agent Loss']
        plt.figure(figsize=(10, 5))
        plt.bar(labels, loss_values)
        plt.xlabel('Loss Type')
        plt.ylabel('Loss Value')
        plt.title('Discriminator Loss')
        plt.savefig('C:\\Users\\Maximilian\\Desktop\\bomberman_rl\\agent_code\\PPO_agent\\discriminator_loss.png')
        plt.close()
            
        
            
    def behavior_cloning(self, expert_data, bc_learning_rate, batch_size):
        save_dir ='C:\\Users\\Maximilian\\Desktop\\bomberman_rl\\agent_code\\PPO_agent'
        #self.actor.train()  # Set the actor network in training mode
        optimizer = optim.Adam(self.actor.parameters(), lr=bc_learning_rate)
        
       
        total_bc_loss = 0.0
        num_samples = len(expert_data)
        print(num_samples)
        
        # Shuffle the expert data for each epoch
        #np.random.shuffle(expert_data) #worse results with it
        
        step_counter = 0#############
        
        for i in range(0, num_samples, batch_size):
            
            if step_counter >= 12:############ 16
                break 
                
            #print(i)
            # Get a batch of expert data
            batch_data = expert_data[i:i+batch_size]
            
            batch_states = []
            batch_actions = []
                
            missed = 0
            next_state_data = 0
            
            for step_entry in batch_data:
                
                step_counter += 1 ##############
                
                expert_state = state_to_features(step_entry.get("state"))
                expert_action_str = step_entry.get("action")
                    
                
                if expert_action_str is None or expert_state is None:
                    continue
                        
                if i < len(expert_data) - 1:
                    next_step_entry = expert_data[i + 1]
                    next_state_data = next_step_entry.get("state")
                
                if next_state_data is None or not next_state_data:
                    break
                
                if expert_action_str is None:
                    missed += 1
                    if missed>10:
                        break
                    continue
            
                    
                # Convert the expert action to a numeric action index
                expert_action = ACTIONS.index(expert_action_str)
                
                batch_states.append(expert_state.flatten())
                batch_actions.append(expert_action)
                    
                
                
            batch_states_array = np.array(batch_states, dtype=np.float32)
            batch_states_tensor = T.tensor(batch_states_array, dtype=T.float).to(self.actor.device)
            
            batch_states_tensor = batch_states_tensor.unsqueeze(0)
                
            #print(features_tensor.shape)
            if batch_states_tensor.shape == T.Size([1, 0]):
                continue
        
            # Forward pass to get agent action probabilities
            action_probs = self.actor(batch_states_tensor)
            
        
            # Convert the predicted actions to a tensor
            predicted_actions = action_probs.probs
            #predicted_actions = action_probs
            
            
            # Convert expert actions to a tensor
            batch_expert_actions_tensor = T.tensor(batch_actions, dtype=T.long).to(self.actor.device)
            size = batch_expert_actions_tensor.size()
            print("size:", size[0])
            predicted_actions = predicted_actions.view(size[0], -1)
        
            
            #predicted_actions_tensor = predicted_actions_tensor.float()
            print("action probs:", action_probs)
            print("predicted_actions:", predicted_actions)
            print("expert_actions:", batch_expert_actions_tensor)
            batch_states_tensor = batch_states_tensor.requires_grad_()
            #predicted_actions_tensor = predicted_actions_tensor.requires_grad_()
            predicted_actions_tensor = predicted_actions.requires_grad_()

            
            # Calculate the behavior cloning loss for the batch
            # Calculate the behavior cloning loss for the batch
            #bc_loss = F.mse_loss(predicted_actions_tensor, batch_expert_actions_tensor)
            bc_loss = F.cross_entropy(predicted_actions_tensor, batch_expert_actions_tensor)
            total_bc_loss += bc_loss.item()
            
            batch_states_tensor = batch_states_tensor.requires_grad_()
            predicted_actions_tensor = predicted_actions_tensor.requires_grad_()

            # Zero the gradients, perform backpropagation, and update the policy
            optimizer.zero_grad()
            bc_loss.backward()
            optimizer.step()
    
        #avg_bc_loss = total_bc_loss / num_samples
        avg_bc_loss = total_bc_loss / (step_counter/batch_size)##################
        print(avg_bc_loss)
        return avg_bc_loss 

    
    #self.actor.eval()  # Set the actor network back to evaluation mode
        # Usage:
        # behavior_cloning(agent, expert_data, num_bc_epochs=100, bc_learning_rate=0.001, batch_size=32)

                
    def conjugate_gradient(self, natural_gradient, surrogate_loss, n_steps=10, residual_tol=1e-10):
        p = natural_gradient.clone().detach()
        r = natural_gradient.clone().detach()

        for _ in range(n_steps):
            z = self.hessian_vector_product(surrogate_loss, p)
            alpha = torch.norm(r)**2 / torch.dot(p, z)
            self.actor.optimizer.zero_grad()
            alpha.backward(retain_graph=True)
            self.actor.optimizer.step()

            r -= alpha * z
            if torch.norm(r) < residual_tol:
                break

            beta = torch.norm(r)**2 / torch.norm(r - alpha * z)**2
            p = r + beta * p

        return p

    def hessian_vector_product(self, surrogate_loss, vector):
        kl = (surrogate_loss * vector).mean()
        grads = torch.autograd.grad(kl, self.actor.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])
        return flat_grad_kl + self.damping * vector

    def learn_gail(self, expert_states_actions):
        for _ in range(self.n_epochs_policy):
            state_arr, action_arr, old_prob_arr, vals_arr, \
                reward_arr, dones_arr, batches = self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (reward_arr[k] + self.gamma * values[k + 1] * \
                                       (1 - int(dones_arr[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t
            advantage = T.tensor(advantage).to(self.actor.device)

            values = T.tensor(values).to(self.actor.device)
            for batch in batches:
                states = state_arr[batch]
                old_probs = T.tensor(old_prob_arr[batch]).to(self.actor.device)

                actions = action_arr[batch]
                numeric_actions = [ACTIONS_NUM[ACTIONS.index(action)] for action in actions]
                actions_tensor = T.tensor(numeric_actions, dtype=T.int64).to(self.actor.device)

                features_tensor = T.from_numpy(states)
                features_tensor = features_tensor.unsqueeze(0)

                dist = self.actor(features_tensor)
                critic_value = self.critic(features_tensor)

                critic_value = T.squeeze(critic_value)

                new_probs = dist.log_prob(actions_tensor)
                prob_ratio = new_probs.exp() / old_probs.exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1 - self.policy_clip,
                                                 1 + self.policy_clip) * advantage[batch]

                # Calculate GAIL cost: -log(D(s, a)) for both expert and generator data
                gail_cost_expert = -self.discriminator(expert_states_actions).log()
                gail_cost_generator = -self.discriminator(np.concatenate(states.flatten(), np.array(actions_tensor))).log()

                # Calculate the Q(s, a) using the discriminator
                q_value_expert = gail_cost_expert.detach()
                q_value_generator = gail_cost_generator.detach()

                # Calculate the advantage
                advantage_expert = q_value_expert - values[batch]
                advantage_generator = q_value_generator - values[batch]

                # Calculate the surrogate loss for the policy
                surrogate_loss = -(T.min(weighted_probs, weighted_clipped_probs) * (advantage_expert + advantage_generator)).mean()

                # Calculate the KL divergence between old and new policies
                kl_divergence = T.mean(old_probs.exp() * (old_probs - new_probs))

                # Calculate the natural gradient direction
                natural_gradient = self.conjugate_gradient(new_probs, surrogate_loss)

                # Perform the KL-constrained natural gradient step
                step_size = T.sqrt(2 * self.kl_constraint / kl_divergence)
                natural_gradient *= step_size
                self.actor.optimizer.zero_grad()
                natural_gradient.backward()
                self.actor.optimizer.step()

                # Update the critic
                self.critic_optimizer.zero_grad()
                critic_loss = (advantage_expert + advantage_generator - critic_value).pow(2).mean()
                critic_loss.backward()
                self.critic_optimizer.step()

        self.memory.clear_memory()


def get_bomb_rad_dict(game_state:dict):
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
    
    game_state = {
        'round': int(game_state['round']),
        'step': int(game_state['step']),
        'field': np.array(game_state['field']),
        'self': tuple(item if not isinstance(item, list) else tuple(item) for item in game_state['self']),
        #'others': [tuple(agent_info) for agent_info in new_game_state['others']],
        'others': [tuple(tuple(inner) if isinstance(inner, list) else inner for inner in others_info) for others_info in game_state['others']],
        #'bombs': [tuple(bomb_info) for bomb_info in new_game_state['bombs']],
        'bombs': [tuple(tuple(inner) if isinstance(inner, list) else inner for inner in bomb_info) for bomb_info in game_state['bombs']],
        'coins': [tuple(tuple(inner) if isinstance(inner, list) else inner for inner in coin_info) for coin_info in game_state['coins']],
        'user_input': game_state['user_input'],
        'explosion_map': np.array(game_state['explosion_map']),
    }
    
    field = game_state["field"]
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