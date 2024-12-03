import random
from random import randint
from collections import deque
import ourhexenv

import torch
from torch import nn
import torch.nn.functional as f

import numpy as np
import math

import yaml

class ReplayMemory():
    def __init__(self, capacity, seed=None) -> None:
        self.memory = deque([],maxlen=capacity)
        if seed is not None:
            random.seed(seed)

    def push(self, transition):
        self.memory.append(transition)
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    '''
        Initialization
    '''
    def __init__(self, observation_space, action_space, hidden=256) -> None:
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(observation_space, hidden)  # Connected layer 1
        self.fc2 = nn.Linear(hidden, action_space)  # Connected layer 2

        self.action_counts = np.zeros(action_space)  # UCB Exploration
        self.state_visits = 0  # UCB Current State Visit Count

    '''
        Forward pass
    '''
    def forward(self, state):
        x = f.relu(self.fc1(state))  # ReLU transformation
        return self.fc2(x)
    
    '''
        Epsilon-greedy modified to UCB, where c is UCB's exploration constant
    '''
    def get_valid_action(self, state, board, c=0.8):
        action_probs = self.forward(state)  # Q-value probabilities for each action
        valid_actions = []  # List of valid actions

        # All valid actions in the current board state
        for i in range(len(action_probs)):
            row = i // 11
            col = i % 11

            if row < 11 and col < 11:
                if board[row][col] == 0:  # Empty space
                    valid_actions.append(i)

        if not valid_actions:
            return action_probs.argmax().item()

        #print(state)
        valid_action_probs = action_probs[valid_actions]
        #return valid_actions[valid_action_probs.argmax().item()]

        # UCB Implementation
        ucb_values = []

        for action in valid_actions:
            q = valid_action_probs[valid_actions.index(action)].item()
            ucb_term = c * math.sqrt(math.log(self.state_visits + 1)/(self.action_counts[action] +1))
            ucb_values.append(q+ucb_term)

        selected_action = valid_actions[np.argmax(ucb_values)]

        self.action_counts[selected_action] += 1
        self.state_visits +=1

        return selected_action


class G13Agent():

    def __init__(self, env, hyperparameter_set, is_training = False) -> None:
        with open('cs272_pa5/hyperparameters.yml', 'r') as file:
            all_hyperparameters_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameters_sets[hyperparameter_set]

        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.mini_batch_size = hyperparameters['mini_batch_size']
        self.epsilon_init = hyperparameters['epsilon_init']
        self.epsilon_decay = hyperparameters['epsilon_decay']
        self.epsilon_min = hyperparameters['epsilon_min']
        self.network_sync_rate = hyperparameters['network_sync_rate']
        self.learning_rate_a = hyperparameters['learning_rate_a']
        self.discount_factor_g = hyperparameters['discount_factor_g']

        self.replay_buffer = ReplayMemory(self.replay_memory_size)
        self.step = 0

        self.loss_fn = nn.MSELoss()
        self.optimizer = None

        self.env = env
        self.is_training = is_training

        self.dqn = DQN(122, 122)
        self.tgt = DQN(122, 122)
        
        if is_training:
            # Update old model if available, otherwise create a new one
            try:
                checkpoint = torch.load('cs272_pa5/g13agent.pt')
                self.dqn.load_state_dict(checkpoint['policy_dqn_state_dict'])
            except:
                pass
            self.tgt.load_state_dict(self.dqn.state_dict())
            step = 0
            self.optimizer = torch.optim.Adam(self.dqn.parameters(),lr=self.learning_rate_a)
        else:
            try:
                checkpoint = torch.load('cs272_pa5/g13agent.pt')
                self.dqn.load_state_dict(checkpoint['policy_dqn_state_dict'])
                self.tgt.load_state_dict(checkpoint['target_dqn_state_dict'])
            except:
                self.tgt.load_state_dict(self.dqn.state_dict())
    
    def select_action(self, observation, reward, termination, truncation, info):
        if info['direction'] == 0:
            agent = 'player_1'
        else:
            agent = 'player_2'
        state = self.env.observe(agent)
        state = torch.tensor(np.concatenate([
            state["observation"].flatten(),  # Flatten the grid
            [state["pie_rule_used"]],  # Include the discrete flag
            ]), dtype=torch.float, device='cpu')
        with torch.no_grad():
            action = self.dqn.get_valid_action(state, self.env.board)
        return action

    def train(self, observation, reward, termination, truncation, info, action) -> None:
        """
        Train the agent on the given experience (observation, reward, etc.)
        """

        # Update target network periodically
        if self.step % self.network_sync_rate == 0:
            self.tgt.load_state_dict(self.dqn.state_dict())

        # Process the state (this is already done in the runner, so you can directly use observation)
        state = torch.tensor(np.concatenate([
            observation["observation"].flatten(),
            [observation["pie_rule_used"]],
        ]), dtype=torch.float, device='cpu')

        # Store the experience in the replay buffer with the action taken
        next_state = torch.tensor(np.concatenate([
            observation["observation"].flatten(),
            [observation["pie_rule_used"]],
        ]), dtype=torch.float, device='cpu')

        self.replay_buffer.push((state, action, reward, next_state, termination))  # Add experience to buffer

        # Sample mini-batch from replay buffer and optimize
        if len(self.replay_buffer) > self.mini_batch_size:
            mini_batch = self.replay_buffer.sample(self.mini_batch_size)
            self.optimize(mini_batch, self.dqn, self.tgt)  # Optimize the Q-network with the mini-batch

        # Decay epsilon for exploration
        self.epsilon_init = max(self.epsilon_min, self.epsilon_init * self.epsilon_decay)

        self.step += 1
        if termination or truncation:
            return True  # End the episode
        return False  # Continue the episode

    '''
        DQN Optimization based on mini-batch
    '''
    def optimize(self, mini_batch, policy_dqn, target_dqn):
        for state, action, reward, next_state, termination in mini_batch:
            if termination:
                target_q = reward
            else:
                with torch.no_grad():
                    target_q = reward + self.discount_factor_g * target_dqn(next_state).max()
            
            current_q = policy_dqn(state)[action]

            loss = self.loss_fn(current_q, target_q)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

'''
    PLACEHOLDERS FROM P4 FOR TESTING CLASS ENVIRONMENT
'''

class MyDumbAgent():
    def __init__(self, env) -> None:
        self.env = env

    def place(self) -> int:
        xVal = randint(0, self.env.board_size - 1)
        yVal = randint(0, self.env.board_size - 1)

        while self.env.board[xVal][yVal] != 0:
            xVal = randint(0, self.env.board_size - 1)
            yVal = randint(0, self.env.board_size - 1)

        return xVal * self.env.board_size + yVal

    def swap(self) -> int:
        return randint(0,1)

    def select_action(self, observation, reward, termination, truncation, info) -> int:
        return self.place()

class MyABitSmarterAgent():
    def __init__(self, env) -> None:
        self.env = env
        self.visited = set()
        self.start = None

    def place(self) -> int:
        if self.start is None:
            return self.begin()
        temp = self.dfs()
        self.start = (temp // self.env.board_size, temp % self.env.board_size)
        return temp

    def swap(self) -> int:
        return randint(0,1)

    def begin(self) -> int:

        xVal = randint(0, self.env.board_size - 1)
        yVal = randint(0, self.env.board_size - 1)

        while self.env.board[xVal][yVal] != 0:
            xVal = randint(0, self.env.board_size - 1)
            yVal = randint(0, self.env.board_size - 1)

        self.start = (xVal, yVal)
        self.visited.add(self.start)

        return xVal * self.env.board_size + yVal

    def dfs(self) -> int:

        steps = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]
        temp = [self.start]
        while temp:
            x,y = temp.pop()
        
            for stepX, stepY in steps:
                newX, newY = x + stepX, y + stepY
                if 0 <= newX < self.env.board_size and 0 <= newY < self.env.board_size and self.env.board[newX][newY] == 0:
                    if (newX, newY) not in self.visited:
                        temp.append((newX, newY))
                        self.visited.add((newX,newY))
                        return newX * self.env.board_size + newY

        return self.adj()

    def adj(self) -> int:
        x, y = self.start
        steps = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]

        for stepX, stepY in steps:
            newX, newY = x + stepX, y + stepY
            if 0 <= newX < self.env.board_size and 0 <= newY < self.env.board_size and self.env.board[newX][newY] == 0:
                if (newX, newY) not in self.visited:
                    self.visited.add((newX, newY))
                    return newX * self.env.board_size + newY

        return self.begin()

    def select_action(self, observation, reward, termination, truncation, info) -> int:
        return self.place()


if __name__ == '__main__':
    observation_space = 121
    action_space = 122
    net = DQN(observation_space,action_space)
    state = torch.randn(1, observation_space)
    output = net(state)
    print(output)