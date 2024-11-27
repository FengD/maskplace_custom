import argparse
import pickle
from collections import namedtuple

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import matplotlib.pyplot as plt

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

import place_env
import torchvision
from place_db import PlaceDB
import time
from tqdm import tqdm
import random
from comp_res import comp_res
from torch.utils.tensorboard import SummaryWriter
from placement_model import MyCNN, MyCNNCoarse, Actor, Critic
from config import config

# set device to cpu or cuda
device = torch.device('cuda')

if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

# Parameters
parser = argparse.ArgumentParser(description='Solve the Pendulum-v0 with PPO')
parser.add_argument('--gamma', type=float, default=0.95, metavar='G', help='discount factor (default: 0.9)')
parser.add_argument('--seed', type=int, default=42, metavar='N', help='random seed (default: 0)')
parser.add_argument('--disable_tqdm', type=int, default=1)
parser.add_argument('--lr', type=float, default=2.5e-3)
parser.add_argument('--log-interval',type=int,default=10,metavar='N',help='interval between training status logs (default: 10)')
parser.add_argument('--pnm', type=int, default=512)
parser.add_argument('--benchmark', type=str, default='adaptec1')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--is_test', action='store_true', default=False)
parser.add_argument('--save_fig', action='store_true', default=False)
parser.add_argument('--tb_log', type=str, default='tb_log', help='the log path of tensorboard')

args = parser.parse_args()

writer = SummaryWriter(args.tb_log)
benchmark = args.benchmark
placedb = PlaceDB(benchmark)
grid = config.grid
placed_num_macro = args.pnm
if args.pnm > placedb.node_cnt:
    placed_num_macro = placedb.node_cnt
    args.pnm = placed_num_macro  
env = gym.make('place_env-v0', placedb = placedb, placed_num_macro = placed_num_macro, grid = grid)

def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

num_action = env.action_space.shape
seed_torch(args.seed)

Transition = namedtuple('Transition',['state', 'action', 'reward', 'a_log_prob', 'next_state', 'reward_intrinsic'])
TrainingRecord = namedtuple('TrainRecord',['episode', 'reward'])
print("seed = {}".format(args.seed))
print("lr = {}".format(args.lr))
print("placed_num_macro = {}".format(args.pnm))

class PPO():
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_epoch = 10
    if placed_num_macro:
        buffer_capacity = 10 * (placed_num_macro)
    else:
        buffer_capacity = 5120
    batch_size = args.batch_size
    print("batch_size = {}".format(batch_size))

    def __init__(self):
        super(PPO, self).__init__()
        self.gcn = None
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.cnn = MyCNN().to(device)
        self.cnn_coarse = MyCNNCoarse(self.resnet, device).to(device)
        self.actor_net = Actor(cnn = self.cnn, gcn = self.gcn, cnn_coarse = self.cnn_coarse).float().to(device)
        self.critic_net = Critic(cnn = self.cnn, gcn = self.gcn,  cnn_coarse = None, res_net = self.resnet).float().to(device)
        self.buffer = []
        self.counter = 0
        self.training_step = 0
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), args.lr)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), args.lr)

    def load_param(self, path):
        checkpoint = torch.load(path, map_location=torch.device(device))
        self.actor_net.load_state_dict(checkpoint['actor_net_dict'])
        self.critic_net.load_state_dict(checkpoint['critic_net_dict'])
    
    def select_action(self, state):
        state = torch.from_numpy(state).float().to(device).unsqueeze(0)
        with torch.no_grad():
            action_probs, _, _ = self.actor_net(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        return action.item(), action_log_prob.item()

    def get_value(self, state):
        state = torch.from_numpy(state)
        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()

    def save_param(self, running_reward):
        strftime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        if not os.path.exists("save_models"):
            os.mkdir("save_models")
        torch.save({"actor_net_dict": self.actor_net.state_dict(),
                    "critic_net_dict": self.critic_net.state_dict()},
                    "./save_models/net_dict-{}-{}-".format(benchmark, placed_num_macro)+strftime+"{}".format(int(running_reward))+".pkl")

    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter+=1
        return self.counter % self.buffer_capacity == 0

    def update(self):
        state = torch.tensor(np.array([t.state for t in self.buffer]), dtype=torch.float)
        action = torch.tensor(np.array([t.action for t in self.buffer]), dtype=torch.float).view(-1, 1).to(device)
        reward = torch.tensor(np.array([t.reward for t in self.buffer]), dtype=torch.float).view(-1, 1).to(device)
        old_action_log_prob = torch.tensor(np.array([t.a_log_prob for t in self.buffer]), dtype=torch.float).view(-1, 1).to(device)
        del self.buffer[:]
        target_list = []
        target = 0
        for i in range(reward.shape[0]-1, -1, -1):
            if state[i, 0] >= placed_num_macro - 1:
                target = 0
            r = reward[i, 0].item()
            target = r + args.gamma * target
            target_list.append(target)
        target_list.reverse()
        target_v_all = torch.tensor(np.array([t for t in target_list]), dtype=torch.float).view(-1, 1).to(device)
       
        for _ in range(self.ppo_epoch): # iteration ppo_epoch 
            for index in tqdm(BatchSampler(SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, True),
                disable = args.disable_tqdm):
                self.training_step +=1
                
                action_probs, _, _ = self.actor_net(state[index].to(device))
                dist = Categorical(action_probs)
                action_log_prob = dist.log_prob(action[index].squeeze())
                ratio = torch.exp(action_log_prob - old_action_log_prob[index].squeeze())
                target_v = target_v_all[index]                
                critic_net_output = self.critic_net(state[index].to(device))
                advantage = (target_v - critic_net_output).detach()

                L1 = ratio * advantage.squeeze() 
                L2 = torch.clamp(ratio, 1-self.clip_param, 1+self.clip_param) * advantage.squeeze() 
                action_loss = -torch.min(L1, L2).mean() # MAX->MIN desent

                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                value_loss = F.smooth_l1_loss(self.critic_net(state[index].to(device)), target_v)
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()

                writer.add_scalar('action_loss', action_loss, self.training_step)
                writer.add_scalar('value_loss', value_loss, self.training_step)


def save_placement(file_path, node_pos, ratio):
    fwrite = open(file_path, 'w')
    node_place = {}
    for node_name in node_pos:

        x, y,_ , _ = node_pos[node_name]
        x = round(x * ratio + ratio) 
        y = round(y * ratio + ratio)
        node_place[node_name] = (x, y)
    print("len node_place", len(node_place))
    for node_name in placedb.node_info:
        if node_name not in node_place:
            continue
        x, y = node_place[node_name]
        fwrite.write('{}\t{}\t{}\t:\tN /FIXED\n'.format(node_name, x, y))
    fwrite.close()
    print(".pl has been saved to {}.".format(file_path))

def check_mk_dir(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

def localtime():
    return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) 

def main():
    agent = PPO()
    
    training_records = []
    running_reward = -1000000

    check_mk_dir("logs")
    log_file_name = "logs/log_"+ benchmark + "_" + localtime() + "_seed_"+ str(args.seed) + "_pnm_" + str(args.pnm) + ".csv"
    fwrite = open(log_file_name, "w")

    load_model_path = "save_models/net_dict-adaptec1-512-2024-11-26-16-43-27-16255.pkl"
    if load_model_path:
       agent.load_param(load_model_path)
    
    best_reward = running_reward
    if args.is_test:
        if not load_model_path:
            print("no model path given for test model")
            return
        score = 0
        state = env.reset()
        done = False
        counter = 0
        while done is False:
            action, action_log_prob = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            assert next_state.shape == (config.num_state, )
            score += reward
            state = next_state
            counter = counter + 1

        print("score = {} counter = {}".format(score, counter))
        torch.inference_mode()
        print("save node_pos")
        hpwl, cost = comp_res(placedb, env.node_pos, env.ratio)
        print("hpwl = {:.2f}\tcost = {:.2f}".format(hpwl, cost))

        strftime_now = localtime()
        check_mk_dir("./gg_place_new")
        pl_file_path = "gg_place_new/{}-{}-{}-{}.pl".format(benchmark, strftime_now, int(hpwl), int(cost)) 
        save_placement(pl_file_path, env.node_pos, env.ratio)

        check_mk_dir("figures")
        env.save_fig("./figures/{}-{}-{}-{}.png".format(benchmark, strftime_now, int(hpwl), int(cost)))
    else:
        for i_epoch in tqdm(range(1000)):
            score = 0
            raw_score = 0
            state = env.reset()

            done = False
            while done is False:
                state_tmp = state.copy()
                action, action_log_prob = agent.select_action(state)
            
                next_state, reward, done, info = env.step(action)
                assert next_state.shape == (config.num_state, )
                reward_intrinsic = 0
                trans = Transition(state_tmp, action, reward / 200.0, action_log_prob, next_state, reward_intrinsic)
                if agent.store_transition(trans):                
                    assert done == True
                    agent.update()
                score += reward
                raw_score += info["raw_reward"]
                state = next_state

            if i_epoch == 0:
                running_reward = score
            running_reward = running_reward * 0.9 + score * 0.1
            # print("score = {}, raw_score = {}".format(score, raw_score))

            if running_reward > best_reward:
                best_reward = running_reward
                if i_epoch >= 100:
                    agent.save_param(running_reward)
                    strftime_now = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
                    pl_file_path = "gg_place_new/{}-{}-{}.pl".format(benchmark, strftime_now, int(raw_score)) 
                    save_placement(pl_file_path, env.node_pos, env.ratio)
                    if args.save_fig:
                        if not os.path.exists("figures"):
                            os.mkdir("figures")
                        env.save_fig("./figures/{}-{}-{}.png".format(benchmark, strftime_now,int(raw_score)))
                        print("save_figure: figures/{}-{}-{}.png".format(benchmark, strftime_now,int(raw_score)))
                    try:
                        print("start try")
                        # cost is the routing estimation based on the MST algorithm
                        hpwl, cost = comp_res(placedb, env.node_pos, env.ratio)
                        print("hpwl = {:.2f}\tcost = {:.2f}".format(hpwl, cost))
                    except:
                        assert False
            
            training_records.append(TrainingRecord(i_epoch, running_reward))
            if i_epoch % 10 ==0:
                print("Epoch {}, Moving average score is: {:.2f} ".format(i_epoch, running_reward))
                fwrite.write("{},{},{:.2f},{}\n".format(i_epoch, score, running_reward, agent.training_step))
                fwrite.flush()
            writer.add_scalar('reward', running_reward, i_epoch)
            if running_reward > -100:
                print("Solved! Moving average score is now {}!".format(running_reward))
                env.close()
                agent.save_param()
                break

        
if __name__ == '__main__':
    main()
