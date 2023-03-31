import environment as env
import random # 에이전트가 무작위로 행동할 확률을 구하기 위해 사용하는 파이썬 기본 패키지입니다.
import math # 에이전트가 무작위로 행동할 확률을 구하기 위해 사용하는 파이썬 기본 패키지입니다.
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import numpy as np

# deque : 선입선출(FIFO) 자료구조의 일종입니다. double-ended queue의 약자로 양쪽 끝에서 삽입과 삭제가 모두 가능합니다.
import matplotlib.pyplot as plt

# 하이퍼파라미터
EPISODES = 100000    # 애피소드 반복횟수
EPS_START = 0.9  # 학습 시작시 에이전트가 무작위로 행동할 확률
EPS_END = 0.05   # 학습 막바지에 에이전트가 무작위로 행동할 확률
EPS_DECAY = 1000000  # 학습 진행시 에이전트가 무작위로 행동할 확률을 감소시키는 값
GAMMA = 0.9      # 할인계수
LR = 0.001       # 학습률
BATCH_SIZE = 32  # 배치 크기


class DQNAgent:
    def __init__(self):
        self.model = nn.Sequential(
            # 대여소 4개의 현재 자전거 개수, 트럭위치 트럭의 자전거 수


            nn.Linear(6, 32),  # 카트 위치, 카트 속도, 막대기 각도, 막대기 속도 4가지 정보를 받습니다(입력노드 4개)
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 17)  # 왼쪽 or 오른쪽으로 출력합니다(출력노드 2개)
        )


        self.optimizer = optim.Adam(self.model.parameters(), LR)  # Adam을 사용해 최적화합니다
        self.steps_done = 0  # 학습을 반복할 때마다 증가하는 변수입니다
        self.memory = deque(maxlen=100000)
        # queue 자료구조를 이용합니다. deque의 maxlen을 지정하면 큐가 가득찼을 때 오래된 요소부터 없어집니다.

    def memorize(self, state, action, reward, next_state, avail_action):
        # memorize(self,현재상태, 현재 상태에서 한 행동, 행동에 대한 보상, 행동으로 인해 새로 생성된 상태):
        if [state,action,torch.FloatTensor([reward]),torch.FloatTensor([next_state])] != []:

            self.memory.append((state,
                            action,
                            torch.FloatTensor([reward]),
                            torch.FloatTensor([next_state]), avail_action))

# memorize availa_action을 추가


    def act(self, state, avail_actions):

        # 초반에는 엡실론 값을 높게하여 최대한 다양한 경험을 해보도록 하고, 점점 그 값을 낮춰가며 신경망이 결정하는 비율을 높임
        # 엡실론 그리디(혹은 엡실론 탐욕) 알고리즘 ( epsilon-greedy )
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if random.random() > eps_threshold:
            avail_actions = torch.tensor(avail_actions, dtype=torch.bool)
            mask = avail_actions.float()
            Q_target = self.model(state)
            Q_target_masked = Q_target.masked_fill(mask == 0, float('-inf'))

            return Q_target_masked.data.max(1)[1].view(1, 1), eps_threshold
        else:

            action_space = [i for i in range(len(avail_actions))]
            avail_actions = np.asarray(avail_actions).astype('float64')
            action = np.random.choice(action_space, p = avail_actions/sum(avail_actions))

            return torch.tensor(action).view(1, 1), eps_threshold



            #return torch.LongTensor([[random.randrange(2)]])

    def learn(self):  # 에이전트가 경험 리플레이를 하며 학습하는 역할을 수행합니다
        if len(self.memory) < BATCH_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)

        #states, actions, rewards, next_states = zip(*batch)  # 4개의 배열로 정리해줍니다
        states, actions, rewards, next_states, avail_actions = zip(*batch)  # 4개의 배열로 정리해줍니다


        states = torch.cat(states)
        actions = torch.cat(actions)
        rewards = torch.cat(rewards)
        next_states = torch.cat(next_states)  # 리스트 형태를 torch.cat() 함수로 하나의 텐서로 만듭니다
        avail_actions = torch.tensor(avail_actions,dtype = torch.bool)
        #print(avail_actions)

        current_q = self.model(states).gather(1, actions)
        mask = avail_actions.float()


        # gather()로 에이전트가 현 상태에서 했던 행동들의 가치를 current_q에 담습니다.
        # self.model(next_states) -> target network(Q_target)
        Q_target = self.model(next_states)
        Q_target_masked = Q_target.masked_fill(mask == 0, float('-inf'))

        #max_next_q = self.model(next_states).detach().max(1)[0]
        #expected_q = rewards + (GAMMA * max_next_q)  # rewards + 미래가치 => 할인된 미래 가치가 expected_q에 담깁니다.

        expected_q = rewards + (GAMMA * Q_target_masked.detach().max(1)[0])

        loss = F.mse_loss(current_q.squeeze(), expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def initialize_memory(self):
        self.memory = [[], [], []]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using ", device)
env = env.Environment(4,30,15) # gym으로 게임환경 생성
agent = DQNAgent()
score_history = []

for e in range(1, EPISODES+1):
    state, avail_action = env.reset() # 게임을 시작할 때마다 게임환경의 상태를 초기화합니다
    steps = 0
    reward_list = list()
    done = False
    while done == False:
        # ex) state2, avail_action2 : action2 => state3, avail_action3라고 하면

        state = torch.FloatTensor([state]) # state2
        action,epsilon = agent.act(state, avail_action) #action2, state2 & avail_action2
        next_state, reward, done, avail_action = env.step(action.item())
        #action2에 대한 reward, done, state3, avail_action3
        agent.memorize(state, action, reward, next_state, avail_action)
        #state2, action2, reward2, state3, avail_action3
        steps += 1
        reward_list.append(reward)

        state = next_state
        #state3로 update


        if done:


            agent.learn()
            #agent.initialize_memory()
            print(e, np.sum(reward_list), epsilon)
            reward_list=[]


        # if not done:
        #     print("에피소드:{0} 점수: {1}".format(e, reward))
        #     score_history.append(steps)
        #     break