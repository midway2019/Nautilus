import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import generrateyaml
import os
from getdata import getprofile
import csv

import argparse
# Hyper Parameters
parser = argparse.ArgumentParser()
parser.add_argument("--bs", type=int, default=32)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--epsilon", type=float, default=0.9)
parser.add_argument("--target", type=int, default=120)
parser.add_argument("--memory", type=int, default=2000)
parser.add_argument("--actions", type=int, default=78)
parser.add_argument("--state", type=int, default=39)
parser.add_argument("--benchmark", type=str, default="socialNetwork")
parser.add_argument("--throughput", type=int, default=50)
parser.add_argument("--qos", type=float, default=1.0)
parser.add_argument("--start_train", type=int, default=32)
parser.add_argument("--theta", type=float, default=0.01)
args = parser.parse_args()

BATCH_SIZE = args.bs
LR = args.lr
EPSILON = args.epsilon
GAMMA = args.gamma
TARGET_REPLACE_ITER = args.target
MEMORY_CAPACITY = args.memory
N_ACTIONS = args.actions
N_STATES = args.state
THETA = args.theta

benchmark = args.benchmark
throughput = args.throughput
qos_target = args.qos
start_train_epoch = args.start_train

class Net(nn.Module):
    def __init__(self,):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(N_STATES,512)
        self.fc1.weight.data.normal_(0,0.1) 
        self.fc2 = nn.Linear(512,256)
        self.fc2.weight.data.normal_(0,0.1) 
        self.fc3 = nn.Linear(256,128)
        self.fc3.weight.data.normal_(0,0.1) 
        self.out = nn.Linear(128,N_ACTIONS)
        self.out.weight.data.normal_(0,0.1) 
        self.V = nn.Linear(128,1)
        self.out.weight.data.normal_(0,0.1) 
    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x,p=0.9)
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x,p=0.9)
        x = self.fc3(x)
        x = F.relu(x)
        x = F.dropout(x,p=0.9)
        v = self.V(x)
        a = self.out(x)
        #actions_value = self.V + self.a
        x3 = v.expand_as(a) + (a - a.mean(1,keepdim=True).expand_as(a))
        actions_value = x3
        #actions_value = self.out(x)
        
        return actions_value

class DQN(object):
    def __init__(self):
        self.eval_net = Net()
        self.target_net = Net()
        self.learn_step_counter = 0 
        self.memory_counter = 0 
        self.memory = np.zeros((MEMORY_CAPACITY,N_STATES * 2 + 2)) 
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(),lr=LR)
        self.loss_func = nn.MSELoss() 

    def caculate_action(self,x):
        x = Variable(torch.unsqueeze(torch.FloatTensor(x),0))
        max_action = 0
        if np.random.uniform() < EPSILON: # greedy
            actions_value = self.eval_net.forward(x)
            actions_value = actions_value.detach().numpy()
            #action = torch.max(actions_value,1)[1].data.numpy()[0]
        else:
            return np.array([-1])
        return actions_value

    def store_transition(self,s,a,r,s_):
        transition = np.hstack((s,[a,r],s_))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index,:] = transition
        self.memory_counter += 1

    def learn(self):
        # target net update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            torch.save(self.eval_net,'net.pkl')
            np.savetxt('memory.csv',self.memory,delimiter=',')
            np.savetxt('learn_step_counter.txt',[self.learn_step_counter,self.memory_counter],delimiter=',')
        self.learn_step_counter = self.learn_step_counter + 1
        sample_index = np.random.choice(MEMORY_CAPACITY,BATCH_SIZE)
        b_memory = self.memory[sample_index,:]
        b_s = Variable(torch.FloatTensor(b_memory[:,:N_STATES]))
        b_a = Variable(torch.LongTensor(b_memory[:,N_STATES:N_STATES+1].astype(int)))
        b_r = Variable(torch.FloatTensor(b_memory[:,N_STATES+1:N_STATES+2]))
        b_s_ = Variable(torch.FloatTensor(b_memory[:,-N_STATES:]))

        q_eval = self.eval_net(b_s).gather(1,b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + GAMMA * q_next.max(1)[0]
        loss = self.loss_func(q_eval,q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def reward(s, a):
    done = False
    resources = np.zeros(s.size)
    if(a < s.size):
        resources[a] = -0.5
        pos = a
    else:
        resources[a % s.size] = 0.5
        pos = a % s.size
    s_ = s + resources
    CPU_usage = 0
    Mem_usage = 0
    IO_usage = 0
    for resources in s_:
        if resources % 3 == 0:
            CPU_usage = CPU_usage + 1
        elif resources % 3 == 1:
            Mem_usage = Mem_usage + 1
        else:
            IO_usage = IO_usage + 1
    CPU_Sum = 40 * 100 / 1000
    Mem_Sum = 90 * 100 / 1000
    IO_Sum = 50 * 100 / 1000
    CPU_usage = CPU_usage/1000
    Mem_usage = Mem_usage/1000
    IO_usage = IO_usage/1000
    CPU_price = 1.054
    Mem_price = 0.55
    IO_price = 1.28
    price = (CPU_Sum / CPU_usage) * CPU_price + (Mem_Sum / Mem_usage) * Mem_price + (IO_Sum / IO_usage) * IO_price
    label = pos % 3
    resource = s_[pos]
    service = int(pos / 3)
    #generrateyaml.changeResources(service,label,resource)
    generrateyaml.generateyaml(deployment, s_, benchmark)
    script = 'sh DeathStarBench/' + benchmark + '/run1.sh ' + str(throughput)
    os.system(script)
    QoS_eval, tr_eval = getprofile(benchmark)
    QoS_target = qos_target
    tr_target = throughput
    latency = QoS_eval/QoS_target
    load = tr_eval/tr_target
    file = open('train_result.csv','a+')
    csv_file = csv.writer(file)
    csv_file.writerow([a,QoS_eval,tr_eval])
    if(latency > 1 or load < 1):
        r = -latency/(load + 10e-6) + THETA * price
    else: 
        r = price
    if tr_eval < throughput:
        done = True
    return s_, r, done
def check_out_resources(s, action, deployment, CPU_Max, Mem_Max, IO_Max):
    a = np.zeros(s.size)
    if(action < s.size):
        a[action] = 0.5
    else:
        a[action % s.size] = -0.5
    s_ = s + a
    if(s_[action % s.size] <= 0):
        return False
    num_of_processes = len(CPU_Max)
    CPU = np.zeros(num_of_processes)
    Mem = np.zeros(num_of_processes)
    IO = np.zeros(num_of_processes)
    for i in range(0, s_.size, 3):
        pos = deployment[int(i/3)]
        CPU[pos] = CPU[pos] + s_[i]
    for i in range(1, s_.size, 3):
        pos = deployment[int(i/3)]
        Mem[pos] = Mem[pos] + s_[i]
    for i in range(2, s_.size, 3):
        pos = deployment[int(i/3)]
        IO[pos] = IO[pos] + s_[i]
    CPU_rest = CPU_Max - CPU
    Mem_rest = Mem_Max - Mem
    IO_rest = IO_Max - IO
    if(action < s.size):
        if((sum(CPU_rest < 0) > 0) or sum(Mem_rest < 0) > 0 or sum(IO_rest < 0) > 0):
            return False
        else:
            return True


def choose_action(state, action, deployment, CPU_Max, Mem_Max, IO_Max):
    if action.size > 1:
        action = action[0]
        max_action, max_pos = np.max(action), np.argmax(action)
        while(check_out_resources(s, max_pos, deployment, CPU_Max, Mem_Max, IO_Max)==False):
            action[max_pos] = -1
            max_action, max_pos = np.max(action), np.argmax(action)
        return max_pos
    else:
        max_pos = np.random.randint(0,2 * state.size)
        while(check_out_resources(s, max_pos, deployment, CPU_Max, Mem_Max, IO_Max)==False):
            max_pos = np.random.randint(0,2 * state.size)
        return max_pos


dqn = DQN()
#deployment = [1,0,0,1,0,0,2,2,0,0,0,1,0]
deployment = [0,0,0,0,0,0,0,0,0,0,0,0,0]
CPU_Max = [40,10,10]
Mem_Max = [90,10,10]
IO_Max = [50,15,15]
print('\nCollecting experience...')

for epoch in range(1000):
    #initialize the enviroment
    s = [3,1,3,2,1,3,2,1,3,2,1,3,3,6,3,2,2,3,4,1,3,2,1,3,1,3,3,1,2,3,2,60,3,1,1,3,8,2,3]
    s = np.array(s,dtype=np.float32)
    script = 'sh DeathStarBench/' + benchmark + '/init.sh'
    os.system(script)
    #s = np.reshape(s.shape[0],1)
    while True:
        a = dqn.caculate_action(s)
        a = choose_action(s, a, deployment, CPU_Max, Mem_Max, IO_Max)
        s_ ,r ,done = reward(s, a)
        #store the dataset
        dqn.store_transition(s,a,r,s_)

        if dqn.memory_counter > start_train_epoch:
            dqn.learn()
        #if done:
            #break
        #set the current state s_ to s
        s = s_
        if done:
            break
'''
for i_episode in range(400):
    s = env.reset()
    while True:
        env.render()
        a = dqn.choose_action(s)
        # take action
        s_,r,done,info = env.step(a)

        # modify the reward
        x,x_dot,theta,theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2

        dqn.store_transition(s,a,r,s_)
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()
        if done:
            break
        s = s_

'''