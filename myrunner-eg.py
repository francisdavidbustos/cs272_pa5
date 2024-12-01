from ourhexenv import OurHexGame
#from gXXagent import GXXAgent
#from gYYagent import GYYAgent
from myagents import MyDumbAgent, MyABitSmarterAgent, DQN
import random

import torch
from torch import nn
import yaml
import numpy as np

from experience_replay import ReplayMemory 


class Agent:
    def __init__(self, hyperparameter_set) -> None:
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

        self.loss_fn = nn.MSELoss()
        self.optimizer = None

    def run(self, is_training=True,render=False):    
        env = OurHexGame(board_size=11,render_mode=None)

        # player 1
        #gXXagent = GXXAgent(env)
        gXXagent = MyDumbAgent(env)
        # player 2
        #gYYagent = GYYAgent(env)
        gYYagent = MyDumbAgent(env)

        #print(env.observation_space("player_1"))
        #print(env.action_space("player_1"))
        #print(env.observation_space("player_2"))
        #print(env.action_space("player_2"))

        num_states = 11*11+1
        num_actions = 122

        rewards_per_episode = []
        epsilon_history = []

        policy_dqn = DQN(num_states,num_actions)

        smart_agent_player_id = random.choice(env.agents)


        replay_memory = ReplayMemory(self.replay_memory_size)
        epsilon = self.epsilon_init
        if is_training:
            target_dqn = DQN(num_states,num_actions)
            target_dqn.load_state_dict(policy_dqn.state_dict())

            step=0

            self.optimizer = torch.optim.Adam(policy_dqn.parameters(),lr=self.learning_rate_a)
        else:
            checkpoint = torch.load('g13agent.pth')
            target_dqn = DQN(num_states,num_actions)
            policy_dqn.load_state_dict(checkpoint['policy_dqn_state_dict'])
            target_dqn.load_state_dict(checkpoint['target_dqn_state_dict'])

        for episode in range(1000):
            env.reset()
            termination = False
            episode_reward = 0.0
            while not termination:
                for agent in env.agent_iter():
                    state = env.observe(agent)
                    state = torch.tensor(np.concatenate([
        state["observation"].flatten(),  # Flatten the grid
        [state["pie_rule_used"]]  # Include the discrete flag
    ]), dtype=torch.float, device='cpu')
                    observation, reward, termination, truncation, info = env.last()
                    
                    if termination or truncation:
                        #print("Terminal state reached")
                        break

                    if is_training and random.random() < epsilon:
                        if agent == 'player_1':
                            action = gXXagent.select_action(observation, reward, termination, truncation, info)
                            #action = gXXagent.env.action_space.sample()
                            action = torch.tensor(action, dtype=torch.int64, device='cpu')
                            action = action.item()
                        else:
                            action = gYYagent.select_action(observation, reward, termination, truncation, info)
                            #action = gYYagent.env.action_space.sample()
                            action = torch.tensor(action, dtype=torch.int64, device='cpu')
                            action = action.item()
                    else:
                        if agent == 'player_1':
                            with torch.no_grad():
                                action = policy_dqn.get_valid_action(state, env.board)
                        else:
                            with torch.no_grad():
                                #action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()
                                action = policy_dqn.get_valid_action(state, env.board)

                    env.step(action)
                    observation, reward, termination, truncation, info = env.last()

                    episode_reward += reward

                    observation = torch.tensor(np.concatenate([
        observation["observation"].flatten(),  # Flatten the grid
        [observation["pie_rule_used"]]  # Include the discrete flag
    ]), dtype=torch.float, device='cpu')
                    reward = torch.tensor(reward, dtype=torch.float, device='cpu')

                    if is_training:
                        replay_memory.push((state, action, observation, reward, termination))
                        step += 1

                    state = observation
                    #env.render()
            rewards_per_episode.append(episode_reward)

            epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
            epsilon_history.append(epsilon)

            if len(replay_memory)>self.mini_batch_size:
                mini_batch = replay_memory.sample(self.mini_batch_size)
                self.optimize(mini_batch, policy_dqn, target_dqn)

                if step > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step = 0
        torch.save({
        'policy_dqn_state_dict': policy_dqn.state_dict(),
        'target_dqn_state_dict': target_dqn.state_dict(),
    }, 'g13agent.pth')

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        for state, action, observation, reward, termination in mini_batch:
            if termination:
                target_q = reward
            else:
                with torch.no_grad():
                    target_q = reward + self.discount_factor_g * target_dqn(observation).max()
            
            current_q = policy_dqn(state)[action]

            loss = self.loss_fn(current_q, target_q)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

if __name__ == '__main__':
    agent = Agent('hexgame1')
    agent.run(is_training=False, render=False)