from ourhexenv import OurHexGame
#from gXXagent import GXXAgent
#from gYYagent import GYYAgent
from myagents import MyDumbAgent, MyABitSmarterAgent, DQN
import random

import torch
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

        if is_training:
            replay_memory = ReplayMemory(self.replay_memory_size)
            epsilon = self.epsilon_init

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

                    state = observation
                    #env.render()
            rewards_per_episode.append(episode_reward)

            epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
            epsilon_history.append(epsilon)

if __name__ == '__main__':
    agent = Agent('hexgame1')
    agent.run(is_training=True, render=False)