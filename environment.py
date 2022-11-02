import os
import time
import psutil
import pickle
import numpy as np
import tensorflow as tf
import json
from RL_AGENT.ddpg import Ddpg
import matplotlib.pyplot as plt
from MODEL.Server.solver import Solver

class Environment(object):
    """
     The Main Environment to connect solver with rl
     How it work:
         For every learning epoch:

             Get last state s (if don't have one, init one)
             Use rl_model.take_action() to generate action pair
             Give action pair to Solver.one_step() to generate new state s_ for rl
             Then calculate reward by env.calculate_reward() get reward r
             Store the transition [s, a, [r], s] by using rl_model.save_transition()

             If action parameter round is i:
                Do Solver.one_step() for i round

             Train the rl with memory set using rl_model.learn
    """
    def __init__(self, dis_dim, scal_dim, s_dim, scal_var, a_bound, s_bound, pow_u, late_u, train=True):
        """
        init solver and rl_model
        :param train: Train rl or not
        :param s_bound: State's boundary if s range is [[0-1],[0-9],[0-100]] then s_bound [0, 9, 100]
        :param pow_u: User's constrain of power
        :param late_u: User's constrain of latency
        :param solver: Solver class, which have .one_step and .build_graph
        :param rl_model: rl_model class, which have .choose_action, .store_transition and .learn
        """
        # da or host 0 = host 1 = da
        self.flag_DA = 1
        self.update_RL = 1
        # init param for rl
        self.dis_dim = dis_dim
        self.scal_dim = scal_dim
        self.s_dim = s_dim
        self.scal_var = scal_var
        self.a_bound = a_bound
        self.s_bound = s_bound
        self.da_memory = []
        # init train and bound
        self.train = train
        # init power and late user constrain
        self.power_u = pow_u
        self.late_u = late_u
        # init solver and rl
        self.solver = Solver()
        print("3")
        self.rl_model = Ddpg(self.dis_dim, self.scal_dim, self.s_dim, self.scal_var, self.a_bound)
        print("Solver Loading")

        if (self.flag_DA == 1):
            da_extractedData = np.load('da.npy', allow_pickle=True).tolist()
            for i in range(0,len(da_extractedData)):
                self.rl_model.store_transition(da_extractedData[i][0],da_extractedData[i][1],da_extractedData[i][2],da_extractedData[i][3])
            print("Restore Domain Adaptation Successfully!")
        # 随机探索的步数
        self.random_step = 1
        # 全局的step
        self.global_step = 0
        self.global_epoch = 0


        self.queue_STATE = []
        self.queue_REWARD = []

        self.init_param_to_store()

    def init_param_to_store(self):
        self.step = []
        self.bandwidth = []
        self.latency = []
        self.power = []
        self.accuracy = []
        self.output_point = []
        self.times = []

        self.epoch = []
        self.bandwidth_epoch = []
        self.latency_epoch = []
        self.power_epoch = []
        self.accuracy_epoch = []
        self.output_point_epoch = []
        self.partition_point = []
        self.threshold = []
        self.round = []
        self.reward = []

        # config:
        self.latency_bound = self.late_u
        self.power_bound = self.power_u
        self.rolling_reward = False
        self.rolling_n = False
        self.fix_partition = False
        self.fix_branch = False
        return

    def store_all(self):
        step = {
            "step": self.step,
            "band_width": self.bandwidth,
            "latency": self.latency,
            "power": self.power,
            "accuracy": self.accuracy,
            "output_point": self.output_point,
            "times": self.times
        }

        epoch = {
            "epoch": self.epoch,
            "band_width": self.bandwidth_epoch,
            "latency": self.latency_epoch,
            "power": self.power_epoch,
            "accuracy": self.accuracy_epoch,
            "output_point": self.output_point_epoch,
            "partition_point": self.partition_point,
            "threshold": self.threshold,
            "round": self.round,
            "reward": self.reward
        }

        config = {
            "latency_bound": self.latency_bound,
            "power_bound": self.power_bound,
            "rolling_reward": self.rolling_reward,
            "rolling_n": self.rolling_n,
            "fix_partition": self.fix_partition,
            "fix_branch": self.fix_branch
        }

        all = {
            "step": step,
            "epoch": epoch,
            "config": config
        }

        file = open("./data.pk", "wb")
        pickle.dump(all, file)
        file.close()



    def init_state(self):
        """
        Init_state if state is empty
        Using mean of s_bound and std of normal generate s_bound
        """
        # calculate mean and std according to s_bound
        print("init_state")
        mean = np.multiply(self.s_bound, 0.5)
        std = np.multiply(self.s_bound, 0.1)
        # generate normal
        state = np.random.normal(mean, std)
        print(state)

        return state

    def sigmoid(self, input):
        return 1 / (1 + np.exp(-input))

    def calculate_p(self, out_point, power, late):
        """
        Calculate penalty according to user constrain
        :param late: real latency
        :param power: real power consumption

        The way to calculate penalty is as follow:
            penalty_power = 0
            penalty_latency = 0

            if power unsatisfied:
                penalty_power = - (power - user_constrain_power)

            if latency unsatisfied:
                penalty_latency = - (latency - user_constrain_latency)
        """

        delta_d = late - self.late_u
        delta_p = power - self.power_u

        if self.power_u < power and self.late_u < late:


            #reward = -1
            reward = -2

        elif self.power_u > power and self.late_u < late:
            reward = - (2 / np.pi) * np.arctan(delta_d)

        elif self.power_u < power and self.late_u > late:
            reward = - (2 / np.pi) * np.arctan(delta_p)

        else:
            #reward = ((out_point+1)/3)**2
            reward = ((out_point+1)/4)**2
        return reward



    def calculate_reward(self, state, out_point):
        """
        Calculate reward of current state
        :param state: [acc, latency, power]

        The way to calculate:
            reward = acc + penalty_power + penalty_latency
            (penalty is calculated using calculate_p() function)
        """
        # state_dim = [acc, latency, power]
        acc, power, late = state[0], state[2], state[1]

        # calculate reward
        reward = self.calculate_p(out_point, power, late)

        return reward

    def take_action(self, action, late_rl):
        """
        Take an action and return a new state
        :param action: [[threshold(n_dim)], cut_point, round_to_jump]
        :param sock: Socket class that solver initialized
        :param sess: Session handel that solver initialized
        :param param: Solver's other parameters

        Used function:
            solver.one_step => a function mapping action to new state
            calculate_reward => a function mapping state to reward

        """
        # 全局step加1，用于标志数据收集
        print("action")
        print(action)
        self.global_epoch += 1

        # actions = [[threshold(n_dim)], cut_point, round_to_jump]
        #if action[3] == 17:
        #    action[3] = 16
        round_to_jump = int(action[4])


        # init next_state, to_average_state
        next_state_to_average = []
        bandwidth_to_avg = []
        out_point_avg = []

        # take one step by solver
        # state_dim = [acc, latency, power]
        print("Small Interval Begin")
        for i in range(round_to_jump+1):
            self.global_step += 1

        #for i in range(1):
            # new state = [network,power]
            # next_state = [acc power latency]
            print("smaill interval {}".format(i))
            #[network], [acc, power, latency], out_point, times
            bandwidth, next_state, out_point, times = self.solver.one_step(action.tolist())
            print(next_state)

            # swap power and latency
            next_state[1], next_state[2] = next_state[2], next_state[1]
            # NOW next_state is [acc, latency, power]

            # 收集相关数据(step)
            self.step.append(self.global_step)
            self.bandwidth.append(bandwidth[0])
            self.latency.append(next_state[1])
            self.power.append((next_state[2]))
            self.accuracy.append(next_state[0])
            self.output_point.append(out_point)
            self.times.append(times)

            # 添加需要average的值
            next_state_to_average.append(next_state)
            bandwidth_to_avg.append(bandwidth)
            # 只有满足的才被加进acc里面做计算
            if self.power_u > next_state[2] and self.late_u > next_state[1]:
                out_point_avg.append(out_point)
            if i == 0:
                next_state[1] += late_rl

        # 计算平均值
        average_state = np.mean(next_state_to_average, axis=0)
        average_bandwidth = np.mean(bandwidth_to_avg, axis=0)
        average_out_point = np.mean(out_point_avg, axis=0)

        # 如果 out_point 为0
        if np.isnan(average_out_point):
            average_out_point = 0

        # calculate reward
        reward = self.calculate_reward(average_state, average_out_point)

        # 收集相关数据:
        self.epoch.append(self.global_epoch)
        self.bandwidth_epoch.append(average_bandwidth[0])
        self.latency_epoch.append(average_state[1])
        self.power_epoch.append(average_state[2])
        self.accuracy_epoch.append(average_state[0])
        self.output_point_epoch.append(average_out_point)
        self.partition_point.append(action[3])
        self.threshold.append(action[0:3])
        self.round.append(action[4])
        self.reward.append(reward)

        # 返回rl的观测数据
        return np.array([average_bandwidth[0], average_state[0], average_state[1], average_state[2]]), reward

    def main(self, epoch):
        """
        Main function of the whole program
        :param epoch: total epoch for rl to run

        How it work:
            For every learning epoch:
                Get last state s (if don't have one, init one)
                Use rl_model.take_action() to generate action pair
                Give action pair to Solver.one_step() to generate new state s_ for rl
                Then calculate reward by env.calculate_reward() get reward r
                Store the transition [s, a, [r], s] by using rl_model.save_transition()
                If action parameter round is i:
                   Do Solver.one_step() for i round
                Train the rl with memory set using rl_model.learn
        """
        # init state at first
        # 3BEI INTERVAL

        state = self.init_state()
        print("State NOrmalization")
        # 初始化随机探索用到的开关
        once = True

        # 初始化用来记录步数的i
        x = []

        key1 = 0
        key2 = 0
        key3 = 0
        key4 = 0
        key5 = 0
        key6 = 0
        key7 = 0
        key8 = 0
        # for every epoch do follow
        for i in range(epoch):
            if(i>1000 and key1 == 0):
                os.system('wondershaper wlp2s0 40 40')
                key1 == 1
            if(i>1500 and key2 == 0):
                os.system('wondershaper clear wlp2s0')
                key2 == 1
            if(i>2000 and key3 == 0):
                os.system('wondershaper wlp2s0 130 130')
                key3 == 1
            if(i>2500 and key4 == 0):
                os.system('wondershaper clear wlp2s0')
                key4 == 1
            if(i>3000 and key5 == 0):
                os.system('wondershaper wlp2s0 40 40')
                key1 == 1
            if(i>3500 and key6 == 0):
                os.system('wondershaper clear wlp2s0')
                key2 == 1
            if(i>4000 and key7 == 0):
                os.system('wondershaper wlp2s0 130 130')
                key3 == 1
            if(i>4500 and key8 == 0):
                os.system('wondershaper clear wlp2s0')
                key4 == 1
            print("EPOCH {} begins".format(i))

            # rl记录时间
            late_rl = time.time()
            # 随机探索
            if (i <= self.random_step):
                if once:
                    self.a_bound = [self.a_bound[1], self.a_bound[2], self.a_bound[3], self.a_bound[4],self.a_bound[5]]
                    once = False
                action = np.clip([np.random.rand() * self.a_bound[0],
                                  np.random.rand() * self.a_bound[1],
                                  np.random.rand() * self.a_bound[2],
                                  np.random.rand() * self.a_bound[3],
                                  np.random.rand() * self.a_bound[4]], np.zeros_like(self.a_bound), self.a_bound)
            # 不随机探索
            else:
                action = self.rl_model.choose_action(state)
                #action = np.array([0.0000000000001,0.00000000000000001,0.00000000000001,0,1])
            # rl完整时间
            late_rl = time.time() - late_rl
            print("Action Choose")
            print(action)
            # take action according to the action, get reward and next_state
            try:
                # np.array([average_bandwidth[0], average_state[0], average_state[1], average_state[2]]), reward
                #[acc, latency, power]
                next_state, reward = self.take_action(action, late_rl)
                print("next state values")
                print(next_state)
            except Exception as e:
                print(e)
                break


            self.queue_STATE.insert(0,state)
            if(len(self.queue_STATE)>3):
                self.queue_STATE.pop()
            state = np.mean(self.queue_STATE, axis=0)

            self.queue_REWARD.insert(0,reward)
            if(len(self.queue_REWARD)>3):
                self.queue_REWARD.pop()
            reward = np.mean(self.queue_REWARD)

            print("Big interval state")
            print(state)
            # store the transition to rl_model memory
            self.rl_model.store_transition(state, action, reward, next_state)
            # store the domain adaptation tragetary
            if(self.update_RL == 1):
                self.da_memory.append([state,action,reward,next_state])

            # if train then train rl
            if self.train:
                self.rl_model.learn()
                if (i % 500 == 0):
                    env.rl_model.save_model_checkpoint()
                    print("I am fucking store you")


            # --optimal: print evaluate
            print_eval = True
            if print_eval:
                # set up evaluation to print
                to_print = "Epoch: {} |" \
                           " State: {} |" \
                           " Action: {} |" \
                           " Reward: {} |" \
                           " Next state: {} |".format(i, state, action, reward, next_state)
                # print no new line or just print
                print(to_print)

            # --optimal: store transitions data for print
            # store = True
            # if store:
            #     # store parameters to visualize
            #     x.append(i)


            # update state

            state = next_state

        # --optional: plot the param
        plot = True
        if plot:
            def plot_data(x_data, y_data, title=' ', x_label=' ', y_label=' ', save=False):
                """plot data"""
                # new figure
                plt.figure()
                # plot, set x, y label and title
                plt.plot(x_data, y_data)
                plt.title(title)
                plt.xlabel(x_label)
                plt.ylabel(y_label)
                # if save then save
                if save:
                    plt.savefig('./{}.png'.format(title))

                # show the fig
                plt.show()
                plt.close(0)

            # plot all data

            """print(self.step)
            print(self.epoch)
            print(self.bandwidth)
            print(self.latency)
            print(self.power)"""

            itemZip = dict(zip(self.epoch,self.bandwidth_epoch))
            with open('results/bandwidth_epoch.json', 'w', encoding='utf-8') as fs:
                json.dump(itemZip, fs)

            itemZip = dict(zip(self.epoch,self.latency_epoch))
            with open('results/latency_epoch.json', 'w', encoding='utf-8') as fs:
                json.dump(itemZip, fs)

            itemZip = dict(zip(self.epoch,self.power_epoch))
            with open('results/power_epoch.json', 'w', encoding='utf-8') as fs:
                json.dump(itemZip, fs)

            itemZip = dict(zip(self.epoch,self.accuracy_epoch))
            with open('results/accuracy_epoch.json', 'w', encoding='utf-8') as fs:
                json.dump(itemZip, fs)

            itemZip = dict(zip(self.epoch,self.output_point_epoch))
            with open('results/output_point_epoch.json', 'w', encoding='utf-8') as fs:
                json.dump(itemZip, fs)

            self.partition_point = [int(tmp) for tmp in self.partition_point]
            #print(self.partition_point)
            #print(type(self.partition_point))
            itemZip = dict(zip(self.epoch,self.partition_point))
            with open('results/partition_point_epoch.json', 'w', encoding='utf-8') as fs:
                json.dump(itemZip, fs)

            self.threshold = [tmp.tolist() for tmp in self.threshold]
            itemZip = dict(zip(self.epoch,self.threshold))
            with open('results/threshold_epoch.json', 'w', encoding='utf-8') as fs:
                json.dump(itemZip, fs)


            self.round = [int(tmp) for tmp in self.round]
            itemZip = dict(zip(self.epoch,self.round))
            with open('results/round_epoch.json', 'w', encoding='utf-8') as fs:
                json.dump(itemZip, fs)

            itemZip = dict(zip(self.epoch,self.reward))
            with open('results/reward_epoch.json', 'w', encoding='utf-8') as fs:
                json.dump(itemZip, fs)

            itemZip = dict(zip(self.step,self.bandwidth))
            with open('results/bandwidth_step.json', 'w', encoding='utf-8') as fs:
                json.dump(itemZip, fs)

            itemZip = dict(zip(self.step,self.latency))
            with open('results/latency_step.json', 'w', encoding='utf-8') as fs:
                json.dump(itemZip, fs)

            itemZip = dict(zip(self.step,self.power))
            with open('results/power_step.json', 'w', encoding='utf-8') as fs:
                json.dump(itemZip, fs)

            itemZip = dict(zip(self.step,self.accuracy))
            with open('results/accuracy_step.json', 'w', encoding='utf-8') as fs:
                json.dump(itemZip, fs)

            itemZip = dict(zip(self.step,self.output_point))
            with open('results/output_point_step.json', 'w', encoding='utf-8') as fs:
                json.dump(itemZip, fs)

            plot_data(self.epoch, self.bandwidth_epoch, title="bandwidth_epoch")
            plot_data(self.epoch, self.latency_epoch, title="latency_epoch")
            plot_data(self.epoch, self.power_epoch, title="power_epoch")
            plot_data(self.epoch, self.accuracy_epoch, title="accuracy_epoch")
            plot_data(self.epoch, self.output_point_epoch, title="output_point_epoch")
            plot_data(self.epoch, self.partition_point, title="partition_point")
            plot_data(self.epoch, self.threshold, title="threshold")
            plot_data(self.epoch, self.round, title="round")
            plot_data(self.epoch, self.reward, title="reward")
            plot_data(self.step, self.bandwidth, title="bandwidth")
            plot_data(self.step, self.latency, title="latency")
            plot_data(self.step, self.power, title="power")
            plot_data(self.step, self.accuracy, title="accuracy")
            plot_data(self.step, self.output_point, title="output_point")
            print("power ")
            print(np.mean(self.power))
            print("accuracy ")
            print(np.mean(self.accuracy_epoch))
            print("accuracy ")
            print(np.mean(self.accuracy))
            print("threshold")
            print(np.mean(self.threshold))
            print("reward ")
            print(np.mean(self.reward))
            print("fill rate")
            print(len([tmp for tmp in self.reward if tmp>0])/len(self.reward))
            print("output_point ")
            print(np.mean(self.output_point))

            #Save DA data to file
            if(self.update_RL == 1):
                da_toFile = np.array(self.da_memory)
                np.save('da.npy',da_toFile )
            #print(da_toFile)
            print("Store Domain Adaptation Successfully!")



if __name__ == '__main__':
    # when use:
    """
    action[4] = interval
    # actions = [[threshold(n_dim)], cut_point, round_to_jump]
    """
    dis_dim = 0
    #[NA. threshold, partition, interval]
    a_bound = [0, 1, 1, 1, 18., 5.]
    #bandwidth [acc, latency, power]
    s_bound = [1, 1, 10, 10]
    scal_dim = np.shape(a_bound[1:])[0]
    s_dim = np.shape(s_bound)[0]
    scal_var = 0.1
    pow_u = 1.5
    late_u = 0.1
    #pow_u = 6.4
    #late_u = 1
    train = True

    # init env
    env = Environment(dis_dim=dis_dim,
                      scal_dim=scal_dim,
                      s_dim=s_dim,
                      scal_var=scal_var,
                      a_bound=a_bound,
                      s_bound=s_bound,
                      pow_u=pow_u,
                      late_u=late_u,
                      train=train)

    # for epoch
    epoch = 5000

    # run main
    env.main(epoch=epoch)
    env.store_all()
    if train:
        env.rl_model.save_model_checkpoint()
