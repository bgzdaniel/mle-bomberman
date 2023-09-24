import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import matplotlib.pyplot as plt
#from .ReplayMemory import ReplayMemory


UP = [0,-1]
RIGHT = [1,0]
DOWN = [0,1]
LEFT = [-1,0]
WAIT = [0,0]

MOVES = {'UP': UP, 'RIGHT': RIGHT, 'DOWN': DOWN, 'LEFT': LEFT, 'WAIT': [0,0], 'BOMB': [0,0]}
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
        plt.savefig('C:\\Users\\Maximilian\\Desktop\\bomberman_rl\\agent_code\\PPO_agent_new_state_to_feature\\discriminator_loss.png')
        plt.close()
        
    
        
    def generate_expert_states_actions(self, expert_data):
        missed = 0
        next_state_data = 0
        
        expert_states_actions = []
        for i, step_entry in enumerate(expert_data):
            state = state_to_features(step_entry.get("state"))
            action_str = step_entry.get("action")
            
            if i < len(data) - 1:
                next_step_entry = data[i + 1]
                next_state_data = next_step_entry.get("state")
                
            if next_state_data is None or not next_state_data:
                break
                
            if action_str is None:
                missed += 1
                if missed>10:
                    action = ACTIONS.index(action_str)
                    expert_states_actions.append(np.concatenate((state.flatten(), np.array([action]))))
                    break
                continue
                
            if action_str is not None:
                action = ACTIONS.index(action_str)
                expert_states_actions.append(np.concatenate((state.flatten(), np.array([action]))))
                
        return expert_states_actions
    
    def generate_agent_states_actions(self, expert_data):
        missed = 0
        next_state_data = 0
        
        agent_states_actions = []
        for i, step_entry in enumerate(expert_data):
            game_state = step_entry.get("state")
            action_str = step_entry.get("action")
                     
            if i < len(data) - 1:
                next_step_entry = data[i + 1]
                next_state_data = next_step_entry.get("state")
            
            if next_state_data is None or not next_state_data:
                break
                
                            
            if action_str is None:
                missed += 1
                if missed>10:
                    break
                continue
           
                
        
            action, _, _ = self.give_back_all(game_state)
            state = state_to_features(game_state)
            agent_states_actions.append(np.concatenate((state.flatten(), np.array([action]))))
        return agent_states_actions

            
    def behavior_cloning(self, expert_data, bc_learning_rate, batch_size):
        save_dir ='C:\\Users\\Maximilian\\Desktop\\bomberman_rl\\agent_code\\PPO_agent_new_state_to_feature'
        #self.actor.train()  # Set the actor network in training mode
        optimizer = optim.Adam(self.actor.parameters(), lr=bc_learning_rate)
        
       
        total_bc_loss = 0.0
        num_samples = len(expert_data)
        print(num_samples)
        
        # Shuffle the expert data for each epoch
        #np.random.shuffle(expert_data) #worse results with it
        
        step_counter = 0#############
        
        for i in range(0, num_samples, batch_size):
            
            #if step_counter >= 12:############ 16
            #    break 
                
            #print(i)
            # Get a batch of expert data
            batch_data = expert_data[i:i+batch_size]
            
            batch_states = []
            batch_actions = []
                
            missed = 0
            next_state_data = 0
            
            for step_entry in batch_data:
                
                step_counter += 1 ##############
                
                expert_state = self.state_to_features(step_entry.get("state"))
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
            
                
                self.last_action = {'move': expert_action_str, 'step': step_entry.get("state")['step']}
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

    def imitation_learning_with_gail(self, expert_states_actions, agent_states_actions):

        # Train discriminator with expert and agent data
        for _ in range(self.epochs_discriminator):################passt noch nicht
            self.learn_discriminator(expert_states_actions, agent_states_actions)

        # Train PPO policy using GAIL feedback
        for _ in range(self.epochs_policy):################passt noch nicht
            self.learn_gail(expert_data)
     
                
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

                # Calculate GAIL cost: -log(D(s, a)) for expert data
                gail_cost_expert = -self.discriminator(states.flatten(), actions_tensor).log()

                # Calculate the Q(s, a) using the discriminator
                q_value = gail_cost_expert.detach()

                # Calculate the advantage
                advantage = q_value - values[batch]

                # Calculate the surrogate loss for the policy
                surrogate_loss = -(T.min(weighted_probs, weighted_clipped_probs) * advantage).mean()

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
                critic_loss = (advantage - critic_value).pow(2).mean()
                critic_loss.backward()
                self.critic_optimizer.step()

        self.memory.clear_memory()
        
    def state_to_features(self, game_state: dict):
        """
            Note: When refactoring chunks of the code, encapsulate it first in a function, add the refactored 
            code in a new function and verify correctness by using asserts and running several rounds
        """
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
    




    """def learn_gail(self, expert_data): #expert_states_actions, agent_states_actions hinzufügen
        for _ in range(self.n_epochs):
            state_arr, action_arr, _, _, _, _, batches = self.memory.generate_batches()

            # Convert action labels to numeric values using ACTIONS_NUM
            numeric_actions = [ACTIONS_NUM[ACTIONS.index(action)] for action in action_arr]
            agent_states_actions = self.generate_agent_states_actions(state_arr, numeric_actions)

            combined_states_actions = np.vstack((expert_states_actions, agent_states_actions))
            labels = np.hstack((np.ones(len(expert_states_actions)), np.zeros(len(agent_states_actions))))

            combined_states_actions = T.tensor(combined_states_actions, dtype=T.float).to(self.actor.device)
            labels = T.tensor(labels, dtype=T.float).to(self.actor.device)

            # Train the discriminator
            discriminator_loss = F.binary_cross_entropy(self.discriminator(combined_states_actions).squeeze(), labels)
            self.discriminator.optimizer.zero_grad()
            discriminator_loss.backward()
            self.discriminator.optimizer.step()

            # Update the policy using discriminator's feedback
            for batch in batches:
                states = state_arr[batch]
                actions = action_arr[batch]
                numeric_actions = [ACTIONS_NUM[ACTIONS.index(action)] for action in actions]
                agent_states_actions = self.generate_agent_states_actions(states, numeric_actions)
                agent_states_actions = T.tensor(agent_states_actions, dtype=T.float).to(self.actor.device)
                advantages = T.tensor(advantage[batch]).to(self.actor.device)

                dist = self.actor(agent_states_actions)
                new_probs = dist.log_prob(actions_tensor)
                prob_ratio = new_probs.exp() / old_probs.exp()
                weighted_probs = advantages * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * advantages
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                # Include the GAIL loss with a weighting factor
                gail_loss = -T.log(self.discriminator(agent_states_actions)).mean()

                total_loss = actor_loss + self.gail_weight * gail_loss

                self.actor.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()"""
                
        
"""def get_bomb_rad_dict(game_state: dict):
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
    return feature_maps"""


    
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

    

