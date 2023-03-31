
from actor_critic import Agent
import numpy as np
import pickle

from environment import Environment

# 대여소수, 대여소 max, 트럭 max

env = Environment(4,30,15)
state_size = 7
action_size = 17
agent = Agent(state_size, action_size)

# with open('agent_4d.pkl', 'rb') as f:
#      agent = pickle.load(f)
#      print("agent_3.load")
for i_episode in range(1001):
    state, avail_actions = env.reset()

    done = False
    t = 0
    rewards = list()
    while done == False:
        t += 1

        if (state[5]==np.argmax(state[:4])and(avail_actions[3]==True)):
            action = 3
        elif(state[5]==np.argmax(state[:4])and(avail_actions[4]==True)):
            action = 4
        elif (state[5] == np.argmax(state[:4]) and (avail_actions[5] == True)):
            action = 5

        elif (state[5] == np.argmin(state[:4]) and (avail_actions[13] == True)):
            action = 13
        elif (state[5] == np.argmin(state[:4]) and (avail_actions[12] == True)):
            action = 12
        elif (state[5] == np.argmax(state[:4]) and (avail_actions[11] == True)):
            action = 11

        else:
            action_space = [i for i in range(len(avail_actions))]
            avail_actions = np.asarray(avail_actions).astype('float64')
            action = np.random.choice(action_space, p=avail_actions / sum(avail_actions))
            #action = 8

        next_state, reward, done, avail_actions = env.step(action)

        rewards.append(reward)

        state = next_state

        if done:

            print("Episode", i_episode, "finished after {} timesteps".format(np.sum(rewards)))
            rewards = []

            if i_episode % 100 == 0:
                with open('agent_4d.pkl', 'wb') as f:
                    pickle.dump(agent, f, protocol=pickle.HIGHEST_PROTOCOL)

            #
            break
