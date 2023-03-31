import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.distributions import Categorical

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using ", device)

class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.fc1 = nn.Linear(state_size, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, action_size)

    def forward(self, state, dim = 0):
        h1 = F.relu(self.fc1(state))
        h2 = F.relu(self.fc2(h1))

        utility = self.fc3(h2)
        pi = F.softmax(utility, dim = dim)
        return pi

class Critic(nn.Module):
    def __init__(self, state_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.fc1 = nn.Linear(state_size, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, state):
        h1 = F.relu(self.fc1(state))
        h2 = F.relu(self.fc2(h1))

        v = self.fc3(h2)
        return v

class Agent():
    def __init__(self, state_size, action_size):
        self.actor = Actor(state_size, action_size)
        self.critic = Critic(state_size)
        self.optim_1 = optim.Adam(self.actor.parameters(), lr = 0.0001)
        self.optim_2 = optim.Adam(self.critic.parameters(), lr = 0.001)
        self.gamma = 0.9
        self.memory = [[],[],[]]


    def get_action(self, state, avail_actions):
        #print(state)
        state = torch.FloatTensor(state)
        #print(state)
        pi = self.actor(state)
        #print(avail_actions)
        #print(pi)

        probs = Categorical(pi*torch.Tensor(avail_actions).float())

        action_tensor = probs.sample()
        action = action_tensor.item()
        #print(action, avail_actions)

        return action

    def _cal_loss(self, states, actions, rewards):
        probs = self.actor(states, dim = 1)
        v_s = self.critic(states)

        actions = actions.long()
        #print(probs.shape, actions.view(-1, 1).shape)
        pi_s = torch.gather(probs, 1, actions.view(-1, 1)).view(-1)
        log_pi = - torch.log(pi_s)
        v_s = v_s.view(-1)

        delta_first = rewards[:-1] + self.gamma * v_s[1:] - v_s[:-1]
        delta_last = (rewards[-1] - v_s[-1]).view(-1)
        advantage = torch.cat((delta_first, delta_last))
        critic_loss = torch.mean(advantage * advantage)
        actor_loss = torch.mean(log_pi*advantage.detach())


        #print(actor_loss)

        return actor_loss, critic_loss

    def train(self):
        states = torch.FloatTensor(self.memory[0])
        actions = torch.LongTensor(self.memory[1])
        rewards = torch.FloatTensor(self.memory[2])
        actor_loss, critic_loss = self._cal_loss(states, actions, rewards)
        #print(loss)
        self.optim_1.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10)
        self.optim_1.step()

        self.optim_2.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10)
        self.optim_2.step()



    def remember(self, state, action, reward):
        self.memory[0].append(state)
        self.memory[1].append(action)
        self.memory[2].append(reward)

    def initialize_memory(self):
        self.memory = [[], [], []]






