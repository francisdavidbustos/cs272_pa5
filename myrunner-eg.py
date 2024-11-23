from ourhexenv import OurHexGame
#from gXXagent import GXXAgent
#from gYYagent import GYYAgent
from myagents import MyDumbAgent, MyABitSmarterAgent
import random

env = OurHexGame(board_size=11)
env.reset()

# player 1
#gXXagent = GXXAgent(env)
gXXagent = MyDumbAgent(env)
# player 2
#gYYagent = GYYAgent(env)
gYYagent = MyDumbAgent(env)

smart_agent_player_id = random.choice(env.agents)

done = False
while not done:
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        
        if termination or truncation:
            print("Terminal state reached")
            break

        
        if agent == 'player_1':
            action = gXXagent.select_action(observation, reward, termination, truncation, info)
        else:
            action = gYYagent.select_action(observation, reward, termination, truncation, info)

        env.step(action)
        env.render()
