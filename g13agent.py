'''
    MOVE EVERYTHING TO DQN
'''

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


class G13Agent:

    '''
        Runs the agent and declares if it is in a testing and sparse environment or not
    '''
    def run(self, is_training=True, sparse=True, render=False):

        # Initialization for visuals and rewards
        if render == False:
            env = OurHexGame(board_size=11, sparse_flag=sparse, render_mode=None)
            if sparse == True:
                print('Sparse rewards')
            else:
                print('Dense rewards')
        else:
            env = OurHexGame(board_size=11, sparse_flag=sparse)
            if sparse == True:
                print('Sparse rewards')
            else:
                print('Dense rewards')


        # player 1
        #gXXagent = GXXAgent(env)
        gXXagent = MyDumbAgent(env)
        # player 2
        #gYYagent = GYYAgent(env)
        gYYagent = MyDumbAgent(env)

        '''
        print(env.observation_space("player_1"))
        print(env.action_space("player_1"))
        print(env.observation_space("player_2"))
        print(env.action_space("player_2"))
        '''

        # Board size is hardcoded, so actions and states are hardcoded
        num_states = 11*11+1
        num_actions = 122

        rewards_per_episode = []
        epsilon_history = []

        policy_dqn = DQN(num_states,num_actions)  # DQN to learn
        target_dqn = DQN(num_states,num_actions)  # DQN to update stability

        replay_memory = ReplayMemory(self.replay_memory_size)  # Store transitions
        epsilon = self.epsilon_init

        if is_training:
            # Update old model if available, otherwise create a new one
            try:
                checkpoint = torch.load('cs272_pa5/g13agent.pt')
                policy_dqn.load_state_dict(checkpoint['policy_dqn_state_dict'])
            except:
                pass
            target_dqn.load_state_dict(policy_dqn.state_dict())
            step = 0
            self.optimizer = torch.optim.Adam(policy_dqn.parameters(),lr=self.learning_rate_a)
        else:
            try:
                checkpoint = torch.load('cs272_pa5/g13agent.pt')
                policy_dqn.load_state_dict(checkpoint['policy_dqn_state_dict'])
                target_dqn.load_state_dict(checkpoint['target_dqn_state_dict'])
            except:
                target_dqn.load_state_dict(policy_dqn.state_dict())

        # Set episode count based on if training or not
        if is_training:
            epr = 1000
        else:
            epr = 1

        for episode in range(epr):
            env.reset()
            termination = False
            episode_reward = 0.0
            while not termination:
                for agent in env.agent_iter():
                    pid = 0 if agent == 'player_1' else 1

                    state = env.observe(agent)
                    state = torch.tensor(np.concatenate([
                        state["observation"].flatten(),  # Flatten the grid
                        [state["pie_rule_used"]],  # Include the discrete flag
                        ]), dtype=torch.float, device='cpu')

                    observation, reward, termination, truncation, info = env.last()
                    
                    if termination or truncation:
                        #print("Terminal state reached")
                        break

                    if is_training and random.random() < epsilon:
                        '''
                        if agent == 'player_1':
                            action = gXXagent.select_action(observation, reward, termination, truncation, info)
                            action = torch.tensor(action, dtype=torch.int64, device='cpu')
                            action = action.item()
                        else:
                            action = gYYagent.select_action(observation, reward, termination, truncation, info)
                            action = torch.tensor(action, dtype=torch.int64, device='cpu')
                            action = action.item()
                        '''
                        action = gXXagent.select_action(observation,reward,termination,truncation,info)
                        action = torch.tensor(action,dtype=torch.int64,device='cpu')
                        action = action.item()
                    else:
                        if agent == 'player_2':
                            action = gXXagent.select_action(observation, reward, termination, truncation, info)
                        else:
                            action = policy_dqn.select_action(observation, reward, termination, truncation, info)
                            with torch.no_grad():
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
        if is_training:
            torch.save({
                'policy_dqn_state_dict': policy_dqn.state_dict(),
                'target_dqn_state_dict': target_dqn.state_dict(),
                }, 'cs272_pa5/g13agent.pt')

if __name__ == '__main__':
    agent = G13Agent('hexgame1')
    agent.run(is_training=True,sparse=True,render=False)
    agent.run(is_training=True,sparse=False,render=False)
    agent.run(is_training=False,sparse=True,render=True)
    agent.run(is_training=False,sparse=False,render=True)