
from actor_critic import Agent
import numpy as np
import pickle

from environment import Environment

# 대여소수, 대여소 max, 트럭 max

env = Environment(4,30,15)
state_size = 6
action_size = 17
agent = Agent(state_size, action_size)

# with open('agent_4d.pkl', 'rb') as f:
#      agent = pickle.load(f)
#      print("agent_3.load")
for i_episode in range(10000):
    state, avail_actions = env.reset()

    done = False
    t = 0
    rewards = list()
    while done == False:
        t += 1

        action = agent.get_action(state, avail_actions)
        #print(state)
        next_state, reward, done, avail_actions = env.step(action)
        agent.remember(state, action, reward)
        rewards.append(reward)

        state = next_state

        if done:
            agent.train()
            agent.initialize_memory()
            print("Episode", i_episode, "finished after {} timesteps".format(np.sum(rewards)))
            rewards = []

            if i_episode % 100 == 0:
                with open('exp2.pkl', 'wb') as f:
                    pickle.dump(agent, f, protocol=pickle.HIGHEST_PROTOCOL)

            #
            break
