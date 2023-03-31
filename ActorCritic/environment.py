import pandas as pd
import numpy as np


rental_05_a = pd.read_csv('2205_319_timetable.csv')
rental_05_b = pd.read_csv('2205_309_timetable.csv')
rental_05_c = pd.read_csv('2205_2961_timetable.csv')
rental_05_d = pd.read_csv('2205_317_timetable.csv')

class Rental():
    def __init__(self, bic_num):
        self.bic_num = bic_num


class Truck():
    def __init__(self, bic_num, loc):
        self.bic_num = bic_num
        self.loc = loc


class Customer():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.delta = int(np.round(np.random.normal(self.mean, self.std, 1)))


class Time():
    def __init__(self, time, clock, day):
        self.time = time
        self.clock = clock
        self.day = day


class Environment():
    def __init__(self, rental_num, rental_max, truck_max):
        self.rental_num = rental_num  # 대여소의 수
        self.rental_max = rental_max  # 각 대여소 최대 보유 자전거 수
        self.truck_max = truck_max  # 트럭에 실을 수 있는 최대 자전거 수

        self.rentals = [Rental(0) for _ in range(self.rental_num)]
        self.truck = Truck(0, 0)
        self.customers = [Customer(1, 0) for _ in range(self.rental_num)]
        self.time = Time(0,0,0)


    def step(self, action):
        reward = 0
        valid = [True, True, True, True, True, True, True,True, True, True, True, True, True, True, True, True, True]

        # 17가지 action 정의

        if action == 0:  # 트럭으로 자전거 15대 옮기기
            self.rentals[self.truck.loc].bic_num -= 15
            self.truck.bic_num += 15
            reward = 1

        if action == 1:  # 트럭으로 자전거 13대 옮기기
            self.rentals[self.truck.loc].bic_num -= 13
            self.truck.bic_num += 13
            reward = 1

        if action == 2:  # 트럭으로 자전거 11대 옮기기
            self.rentals[self.truck.loc].bic_num -= 11
            self.truck.bic_num += 11
            reward = 1

        if action == 3:  # 트럭으로 자전거 9대 옮기기
            self.rentals[self.truck.loc].bic_num -= 9
            self.truck.bic_num += 9
            reward = 1

        if action == 4:  # 트럭으로 자전거 7대 옮기기
            self.rentals[self.truck.loc].bic_num -= 7
            self.truck.bic_num += 7
            reward = 1

        if action == 5:  # 트럭으로 자전거 5대 옮기기
            self.rentals[self.truck.loc].bic_num -= 5
            self.truck.bic_num += 5
            reward = 1

        if action == 6:  # 트럭으로 자전거 3대 옮기기
            self.rentals[self.truck.loc].bic_num -= 3
            self.truck.bic_num += 3
            reward = 1

        if action == 7:  # 트럭으로 자전거 1대 옮기기
            self.rentals[self.truck.loc].bic_num -= 1
            self.truck.bic_num += 1
            reward = 1

        if action == 8:  # 아무것도 안하기
            reward = 1

        if action == 9:  # 트럭에서 자전거 1대 옮기기
            self.rentals[self.truck.loc].bic_num += 1
            self.truck.bic_num -= 1
            reward = 1
        if action == 10:  # 트럭에서 자전거 3대 옮기기
            self.rentals[self.truck.loc].bic_num += 3
            self.truck.bic_num -= 3
            reward = 1
        if action == 11:  # 트럭에서 자전거 5대 옮기기
            self.rentals[self.truck.loc].bic_num += 5
            self.truck.bic_num -= 5
            reward = 1
        if action == 12:  # 트럭에서 자전거 7대 옮기기
            self.rentals[self.truck.loc].bic_num += 7
            self.truck.bic_num -= 7
            reward = 1
        if action == 13:  # 트럭에서 자전거 9대 옮기기
            self.rentals[self.truck.loc].bic_num += 9
            self.truck.bic_num -= 9
            reward = 1
        if action == 14:  # 트럭에서 자전거 11대 옮기기
            self.rentals[self.truck.loc].bic_num += 11
            self.truck.bic_num -= 11
            reward = 1
        if action == 15:  # 트럭에서 자전거 13대 옮기기
            self.rentals[self.truck.loc].bic_num += 13
            self.truck.bic_num -= 13
            reward = 1
        if action == 16:  # 트럭에서 자전거 15대 옮기기
            self.rentals[self.truck.loc].bic_num += 15
            self.truck.bic_num -= 15
            reward = 1

        # state 정의
        state = [self.rentals[i].bic_num for i in range(self.rental_num)]

        before_truck_loc = self.truck.loc
        # 트럭 위치 변화 (순환)
        self.truck.loc += 1
        if self.truck.loc > (self.rental_num - 1):
            self.truck.loc -= self.rental_num


        # 시간 증가, 1000시간 되면 종료
        self.time.time += 1
        if self.time.time == 1000:
            done = True
        else:
            done = False
            
        # clock & day
        # ex) 26시간이면 2clock & 1day
        clock = int(self.time.time % 24)
        day = int(self.time.time // 24)
        day_num = day

        while day_num > len(rental_05_a.columns) - 1:
            day_num -= len(rental_05_a.columns)


        # 실 데이터 적용
        # 대여반납 발생

        # self.customers[0].mean = rental_05_a[rental_05_a.columns[day_num]][clock]
        # self.customers[1].mean = rental_05_b[rental_05_b.columns[day_num]][clock]
        # self.customers[2].mean = rental_05_c[rental_05_c.columns[day_num]][clock]/6
        # self.customers[3].mean = rental_05_d[rental_05_d.columns[day_num]][clock]

        # 각 대여소 손님의 대여와 반납 생성
        for i in range(self.rental_num):
            self.customers[i].delta = int(np.round(np.random.normal(self.customers[i].mean, self.customers[i].std, 1)))
        
        # 음수가 되면 -reward 부여, 0으로 수 조정
        for i in range(self.rental_num):
            self.rentals[i].bic_num = self.customers[i].delta + self.rentals[i].bic_num
            if self.rentals[i].bic_num < 0:
                #reward -= 6.0
                reward += self.rentals[i].bic_num
                #print("balking!!")
            self.rentals[i].bic_num = max(self.rentals[i].bic_num, 0)
        
        # 대여소 max인 30을 넘으면 -reward부여, 30으로 수 조정
        for i in range(self.rental_num):
            if self.rentals[i].bic_num > 30:
                #reward -= 3.0
                reward -= (self.rentals[i].bic_num-30)
            self.rentals[i].bic_num = min(self.rentals[i].bic_num, 30)

        state = [self.rentals[i].bic_num for i in range(self.rental_num)]
        
        next_state = [self.rentals[i].bic_num / self.rental_max for i in range(self.rental_num)]
        next_state.extend([self.truck.bic_num / self.truck_max, self.truck.loc / (self.rental_num - 1)])
        
        
        
        #불가능한 action valid 조정
        if (self.truck.bic_num > self.truck_max - 15) or (self.rentals[self.truck.loc].bic_num < 15):
            valid[0] = False
        if (self.truck.bic_num > self.truck_max - 13) or (self.rentals[self.truck.loc].bic_num < 13):
            valid[1] = False
        if (self.truck.bic_num > self.truck_max - 11) or (self.rentals[self.truck.loc].bic_num < 11):
            valid[2] = False
        if (self.truck.bic_num > self.truck_max - 9) or (self.rentals[self.truck.loc].bic_num < 9):
            valid[3] = False
        if (self.truck.bic_num > self.truck_max - 7) or (self.rentals[self.truck.loc].bic_num < 7):
            valid[4] = False
        if (self.truck.bic_num > self.truck_max - 5) or (self.rentals[self.truck.loc].bic_num < 5):
            valid[5] = False
        if (self.truck.bic_num > self.truck_max - 3) or (self.rentals[self.truck.loc].bic_num < 3):
            valid[6] = False
        if (self.truck.bic_num > self.truck_max - 1) or (self.rentals[self.truck.loc].bic_num < 1):
            valid[7] = False

        if (self.truck.bic_num < 1) :
            valid[9] = False
        if (self.truck.bic_num < 3):
            valid[10] = False
        if (self.truck.bic_num < 5) :
            valid[11] = False
        if (self.truck.bic_num < 7):
            valid[12] = False
        if (self.truck.bic_num < 9):
            valid[13] = False
        if (self.truck.bic_num < 11):
            valid[14] = False
        if (self.truck.bic_num < 13):
            valid[15] = False
        if (self.truck.bic_num < 15):
            valid[16] = False


        return next_state, reward/10000.0 , done, valid


    def reset(self):

        # 각 대여소 초기 자전거 개수
        # 각 대여소 승객 분포 (평균, 표준편차)

        self.customers[0].mean = 2
        self.customers[1].mean = -1
        self.customers[2].mean = 1
        self.customers[3].mean = -2

        for i in range(self.rental_num):
            self.rentals[i].bic_num = 7
            self.customers[i].std = 0

            self.customers[i].delta = int(np.round(np.random.normal(self.customers[i].mean, self.customers[i].std, 1)))

        # 시간, 초기트럭보유자전거수, 초기위치 : -1
        self.time = Time(0,0,0)
        self.truck.bic_num = 10
        self.truck.loc = -1

        # 초기 state
        state = [self.rentals[i].bic_num for i in range(self.rental_num)]


        # 초기 시간 증가와 대여반납 발생
        self.time.time += 1
        for i in range(self.rental_num):
            self.rentals[i].bic_num = min(self.customers[i].delta + self.rentals[i].bic_num, self.rental_max)
        state = [self.rentals[i].bic_num/self.rental_max for i in range(self.rental_num)]

        # 트럭 1대여소 도착
        self.truck.loc += 1
        state.extend([self.truck.bic_num/self.truck_max, self.truck.loc/(self.rental_num-1)])







        valid = [True, True, True, True, True, True, True,True, True, True, True, True, True, True, True, True, True]
        if (self.truck.bic_num > self.truck_max - 15) or (self.rentals[self.truck.loc].bic_num < 15):
            valid[0] = False
        if (self.truck.bic_num > self.truck_max - 13) or (self.rentals[self.truck.loc].bic_num < 13):
            valid[1] = False
        if (self.truck.bic_num > self.truck_max - 11) or (self.rentals[self.truck.loc].bic_num < 11):
            valid[2] = False
        if (self.truck.bic_num > self.truck_max - 9) or (self.rentals[self.truck.loc].bic_num < 9):
            valid[3] = False
        if (self.truck.bic_num > self.truck_max - 7) or (self.rentals[self.truck.loc].bic_num < 7):
            valid[4] = False
        if (self.truck.bic_num > self.truck_max - 5) or (self.rentals[self.truck.loc].bic_num < 5):
            valid[5] = False
        if (self.truck.bic_num > self.truck_max - 3) or (self.rentals[self.truck.loc].bic_num < 3):
            valid[6] = False
        if (self.truck.bic_num > self.truck_max - 1) or (self.rentals[self.truck.loc].bic_num < 1):
            valid[7] = False

        if (self.truck.bic_num < 1) :
            valid[9] = False
        if (self.truck.bic_num < 3):
            valid[10] = False
        if (self.truck.bic_num < 5) :
            valid[11] = False
        if (self.truck.bic_num < 7):
            valid[12] = False
        if (self.truck.bic_num < 9):
            valid[13] = False
        if (self.truck.bic_num < 11):
            valid[14] = False
        if (self.truck.bic_num < 13):
            valid[15] = False
        if (self.truck.bic_num < 15):
            valid[16] = False


        return state, valid



