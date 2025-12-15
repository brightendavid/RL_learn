import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from gridenv import grid_env

"""
这是基于神经网络的方法，结合深度学习和强化学习
主要是因为q-learn是off policy的方法，深度学习需要预先准备数据集，不可能使用on policy的方法。
采用on policy方法迭代一次采样一次的速度太慢，感觉硬来也可以实现

dqn

"""
# 定义网络，输入是S，A ，S‘  输出是概率P
# 三进一出
class QNET(nn.Module):
    def __init__(self, input_dim=3, output_dim=1):
        super(QNET, self).__init__()
        self.fc =  nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=output_dim)
        )
    def forward(self, x):
        x = x.type(torch.float32)
        return self.fc(x)

class Solve:
    def __init__(self, env: grid_env.GridEnv):
        self.gama = 0.9
        self.env = env
        self.action_space_size = env.action_space_size
        self.state_space_size = env.size ** 2
        self.reward_space_size, self.reward_list = len(self.env.reward_list), self.env.reward_list
        self.state_value = np.zeros(shape=self.state_space_size)
        self.qvalue = np.zeros(shape=(self.state_space_size, self.action_space_size))
        self.mean_policy = np.ones(shape=(self.state_space_size, self.action_space_size)) / self.action_space_size
        self.policy = self.mean_policy.copy()

    def show_policy(self):
        # 展示现有policy，就是self.policy
        # env.render_就是Render在env的实例化
        for state in range(self.state_space_size):
            for action in range(self.action_space_size):
                policy = self.policy[state,action] # policy 读取样式
                self.env.render_.draw_action(
                    pos = self.env.state2pos(state),
                    toward=policy*0.4*self.env.action_to_direction[action],
                    radius= policy*0.1 # radius,控制action 5 原地不动的圆的大小R半径
                )

    def obtain_episode(self, policy, start_state, start_action, length):
        """
        这个东西是不停的呀！
        可以一直生成下去，长度为length
        :param policy: 由指定策略产生episode
        :param start_state: 起始state
        :param start_action: 起始action
        :param length: episode 长度
        :return: 一个 state,action,reward,next_state,next_action 序列
        """
        self.env.agent_location = self.env.state2pos(start_state)
        episode = []
        next_action = start_action
        next_state = start_state
        while length > 0:
            length -= 1
            state = next_state
            action = next_action
            _, reward, done, _, _ = self.env.step(action)
            next_state = self.env.pos2state(self.env.agent_location)
            next_action = np.random.choice(np.arange(len(policy[next_state])),
                                           p=policy[next_state])
            episode.append({"state": state, "action": action, "reward": reward, "next_state": next_state,
                            "next_action": next_action})
        return episode
    def get_data_iter(self,episode,batch_size=64,is_train =True):
        # Pytorch的迭代器
        # 就是正常网络训练的dataloader部分
        reward = []
        state_action = []
        next_state = []
        for i in range(len(episode)):
            reward.append(episode[i]["reward"])
            aciton = episode[i]["action"]
            y,x = self.env.state2pos(episode[i]["state"])
            # 获取episode的位置
            state_action.append((y,x,aciton)) #  S,a
            y,x = self.env.state2pos(episode[i]["next_state"]) # 这里使用的是pos形式
            next_state.append((y,x)) # s'
        # print(len(reward))  # now type is list
        reward = torch.tensor(reward).reshape(-1,1) #list 转tensor，是（1,50），再reshape是（50,1）
        # print(reward.shape)
        state_action = torch.tensor(state_action) # 转张量
        next_state = torch.tensor(next_state)
        # print(state_action)
        # print(state_action.shape) # (50,3) ((y,x,aciton))

        data_arrays = (state_action,reward,next_state)
        dataset = data.TensorDataset(*data_arrays) # 加* 解包就是去括号
        # 返回的是一个数据集读取器
        return data.DataLoader(dataset, batch_size=batch_size,
                               shuffle=is_train,drop_last=False)
        # shuffle:是否打乱顺序。
        # drop_last:是否放弃最后一个





    def dqn(self,learning_rate = 0.0015,episode_length =5000,
            epochs = 720,batch_size=50,update_step =20):
        """
        dqn,基于off policy的q-learning.
        两个网络，可以看做是判别器和生成器的关系，实际上并不是。
        这个网络训练需要的数据量和epoch比想象的要多。理想情况下，这个loss应当非常小才算比较好的策略。
        计划：1.可以看一看GAN网络的写法。
            2.这些超参数可能设计的有些问题，出来的policy不是很好。
            3.可以整一个save best model模块
        :param learning_rate:学习率
        :param episode_length: 采用长度，就是数据集大小
        :param epochs: 训练迭代次数
        :param batch_size: batch size，多个batch一起训练，100可能太大，这种上百的batch size闻所未闻，太可怕了。batch size太大不好，一般网络设置为8,16就行了
        :param update_step: 和target-q-net同步的step
        :return:
        """
        q_net = QNET()
        policy = self.policy.copy()
        state_value  =self.state_value.copy()
        q_target_net =QNET()
        q_target_net.load_state_dict(q_net.state_dict())
        # 复制q_net的权重
        optimizer = torch.optim.SGD(q_net.parameters(), lr=learning_rate) # 自带的优化器
        # 优化器选择SGD 随机梯度下降，换成Adam也一样
        episode = self.obtain_episode(self.mean_policy,0,
                                      0,length= episode_length)
        data_iter= self.get_data_iter(episode,batch_size)
        # 这是一个数据集读取器
        loss = torch.nn.MSELoss()
        # 交叉熵
        approximation_q_value = np.zeros(shape=(self.state_space_size, self.action_space_size))
        # shape = (25,5) q-table
        i=0
        rmse_list =[]
        loss_list = []
        for epoch in range(epochs):
            if epoch % 100 == 0:
                self.env.rend_reset()
                self.show_policy()
                self.env.render_save()
            for i,(state_action, reward, next_state) in enumerate(data_iter):
                # 读数据,按照batchsize的大小一次去读取的，从数据集中采样
                # i+=1 # 记录当前
                q_value = q_net(state_action)
                # 包含state_pos,action 的一个三元
                # q_net input is 3,output is 1
                # in fact,input:(batch_size,3);out_put:(batch_size,1)
                q_value_target = torch.empty((batch_size,0))
                # target   (batch，0) 第二维没有任何数据，起到一个占位的作用
                for action in  range(self.action_space_size):
                    s_a = torch.cat((next_state,
                                     torch.full((batch_size,1), action)),dim=1)
                    # 拼接next_state,aciton，构造一个(batch_size ,3) 大小的网络输入张量
                    # torch.full(size , num ) 创造一个大小为(batch_size,1)填充数值为action的tensor
                    q_value_target = torch.cat((q_value_target,
                                                q_target_net(s_a)),dim=1)
                    # 将for内产生的所有q_value_target堆叠,叠加维度为第1维
                    # 就是一个（batch_size,1*5）大小的q-table
                q_star = torch.max(q_value_target,dim=1,keepdim =True)[0]
                # 经典还是求最大，最优的aciton
                y_target_value = reward +self.gama * q_star
                # 这是贝尔曼最优公式
                l = loss(q_value,y_target_value)
                # 求loss
                optimizer.zero_grad()
                # 梯度清0
                l.backward()
                optimizer.step()
                # 优化参数

                if i%update_step == 0 and i!=0:
                    q_net.load_state_dict(q_net.state_dict())
                    # 每update_step次迭代，更新target网络参数
                    # 这个思想根源可以追溯到GAN网络的判别器和生成器
                    # 防止一个跑一个追，训练不好的情况.update_step这个超参数需要根据经验设置，考虑网络大小，数据集大小等因素

            loss_list.append(float(l))
            print("loss:{},epoch:{}".format(float(l),epoch))

            self.policy =np.zeros(shape=(self.state_space_size, self.action_space_size))
            # 25*5
            self.state_value = np.zeros(shape = self.state_space_size)
            # 25*1
            # 这个for循环是更新policy
            for s in range(self.state_space_size):
                y,x = self.env.state2pos(s)
                for a in range(self.action_space_size):
                    approximation_q_value[s,a] = float(q_net(torch.tensor((y,x,a)).reshape(-1,3)))
                    # 现在不是张量了
                q_star_index = approximation_q_value[s].argmax()
                # 最优策略，这里的policy只是q_net参数的映射而已
                # 和网络的参数更新是无关的
                self.policy[s,q_star_index] = 1
                self.state_value[s] = approximation_q_value[s,q_star_index]
            rmse_list.append(float(np.sqrt(np.mean(self.state_value-state_value)**2)))
        # fig_rmse =plt.figure(figsize =(8,12))
        # ax_rmse = fig_rmse.add_subplot(211)
        #
        # ax_rmse.plot(rmse_list)
        # ax_rmse.set_title('rmse')
        # ax_loss = fig_rmse.add_subplot(212)
        # ax_loss.plot(loss_list)
        # ax_loss.set_title('loss')
        # ax_loss.set_xlabel('epoch')
        # ax_loss.set_ylabel('loss')
        # plt.show()


if __name__ == '__main__':
    env = grid_env.GridEnv(size=5, target=[2, 3],
                           forbidden=[[2, 2], [2, 1], [1, 1], [3, 3], [1, 3], [1, 4]],
                           render_mode='')  # 初始化环境  size,target,forbidden的初始化
    solver = Solve(env)
    # env.render()  # 显示网格世界
    # episode = solver.obtain_episode(solver.mean_policy, 0,
    #                               0, length=50)
    # print(episode) #{'state': 0, 'action': 0, 'reward': -10, 'next_state': np.int64(0), 'next_action': np.int64(2)}
    # solver.get_data_iter(episode)
    solver.dqn()
    solver.show_policy()
    env.render()