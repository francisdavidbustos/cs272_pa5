from ourhexenv import OurHexGame
from g13agent_refactor import G13Agent
import random
import torch

env = OurHexGame(board_size=11,sparse_flag=False,render_mode=None)
env = OurHexGame(board_size=11,sparse_flag=True)
env.reset()

# player 1
gXXagent = G13Agent(env, 'hexgame1',is_training=False)
# player 2
gYYagent = G13Agent(env,'hexgame1',is_training=False)

smart_agent_player_id = random.choice(env.agents)
episode_count = 10000

done = False
if gYYagent.is_training:
    for episode in range(episode_count):
        done = False
        while not done:
            for agent in env.agent_iter():
                observation, reward, termination, truncation, info = env.last()
                '''
                print(observation)
                print(reward)
                print(termination)
                print(truncation)
                print(info)
                '''
                
                if termination or truncation:
                    done = True
                    break

                
                if agent == 'player_1':
                    action = gXXagent.select_action(observation, reward, termination, truncation, info)
                else:
                    action = gYYagent.select_action(observation, reward, termination, truncation, info)
                    gYYagent.train(observation,reward,termination,truncation,info,action)

                env.step(action)
            torch.save({
                'policy_dqn_state_dict': gYYagent.dqn.state_dict(),
                'target_dqn_state_dict': gYYagent.tgt.state_dict(),
                }, f'cs272_pa5/g13agent_p2.pth')
            torch.save({
                'policy_dqn_state_dict': gXXagent.dqn.state_dict(),
                'target_dqn_state_dict': gXXagent.tgt.state_dict(),
                }, f'cs272_pa5/g13agent_p1.pth')
        print(f"Model saved after episode {episode}")
else:
    while not done:
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            print(observation)
            print(reward)
            print(termination)
            print(truncation)
            print(info)
            
            if termination or truncation:
                break

            
            if agent == 'player_1':
                action = gXXagent.select_action(observation, reward, termination, truncation, info)
            else:
                action = gYYagent.select_action(observation, reward, termination, truncation, info)

            env.step(action)
            env.render()