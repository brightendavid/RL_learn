import random
import time

import matplotlib.pyplot as plt
import numpy as np

import gridenv.render as render

from gridenv import grid_env

"""
VFA方法：（value function approximation）
这个就是和最小二乘拟合的相似的方法，使用函数化的方法去拟合qvalue
最小二乘拟合是y=ax+b去拟合(x,y)的序列
此处输入为s,a，输出为q.
请注意，并不一定要使用神经网络的方式去做
"""

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

    def random_greed_policy(self):
        """
        生成随机的greedy策略
        :return:
        """
        policy = np.zeros(shape=(self.state_space_size, self.action_space_size))
        for state_index in range(self.state_space_size):
            action = np.random.choice(range(self.action_space_size))
            policy[state_index, action] = 1
        return policy

    def policy_evaluation(self, policy, tolerance=0.001, steps=10):
        """
        迭代求解贝尔曼公式 得到 state value tolerance 和 steps 满足其一即可
        :param policy: 需要求解的policy
        :param tolerance: 当 前后 state_value 的范数小于tolerance 则认为state_value 已经收敛
        :param steps: 当迭代次数大于step时 停止计算 此时若是policy iteration 则算法变为 truncated iteration
        :return: 求解之后的收敛值
        """
        state_value_k = np.ones(self.state_space_size)
        state_value = np.zeros(self.state_space_size)
        while np.linalg.norm(state_value_k - state_value, ord=1) > tolerance:
            state_value = state_value_k.copy()
            for state in range(self.state_space_size):
                value = 0
                for action in range(self.action_space_size):
                    value += (policy[state, action]
            * self.calculate_qvalue(state_value=state_value_k.copy(),
                        state=state,action=action))  # bootstrapping
                state_value_k[state] = value
        return state_value_k
    def show_policy(self):
        for state in range(self.state_space_size):
            for action in range(self.action_space_size):
                policy = self.policy[state, action]
                self.env.render_.draw_action(pos=self.env.state2pos(state),
                                             toward=policy * 0.4 * self.env.action_to_direction[action],
                                             radius=policy * 0.1)

    def show_state_value(self, state_value, y_offset=0.2):
        for state in range(self.state_space_size):
            self.env.render_.write_word(pos=self.env.state2pos(state),
                    word=str(np.round(state_value[state], 1)),
                                        y_offset=y_offset,
                                        size_discount=0.7)
    def calculate_qvalue(self, state, action, state_value):
        """
        计算qvalue elementwise形式
        :param state: 对应的state
        :param action: 对应的action
        :param state_value: 状态值
        :return: 计算出的结果
        """
        qvalue = 0
        for i in range(self.reward_space_size):
            qvalue += (self.reward_list[i] *
                       self.env.Rsa[state, action, i])
        for next_state in range(self.state_space_size):
            qvalue += (self.gama * self.env.Psa[state, action, next_state]
                       * state_value[next_state])
        return qvalue

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

    def gfv(self, fourier: bool, state: int, ord: int) -> np.ndarray:
        """
        get_feature_vector
        强化学习特征构造函数，通常用于将低维的离散状态（如网格世界的坐标）
        映射到高维的特征空间，
        以便线性函数逼近器（如线性回归、线性价值函数近似）
        能够更好地拟合复杂的策略或价值函数。

        :param fourier: 是否使用傅里叶特征函数
        :param state: 状态
        :param ord: 特征函数最高阶次数/傅里叶q(对应书)
        :return: 代入state后的计算结果,shape is (ord+1)**2
        """

        if state < 0 or state >= self.state_space_size:
            raise ValueError("Invalid state value")
        y, x = self.env.state2pos(state) + (1, 1)
        feature_vector = []
        if fourier:
            # 归一化到 -1 到 1
            x_normalized = x / self.env.size
            y_normalized = y / self.env.size
            for i in range(ord + 1):
                for j in range(ord + 1):
                    feature_vector.append(np.cos(np.pi * (i * x_normalized + j * y_normalized)))

        else:
            # 归一化到 0 到 1
            x_normalized = (x - (self.env.size - 1) * 0.5) / (self.env.size - 1)
            y_normalized = (y - (self.env.size - 1) * 0.5) / (self.env.size - 1)
            for i in range(ord + 1):
                for j in range(i + 1):
                    feature_vector.append(y_normalized ** (ord - i) * x_normalized ** j)

        return np.array(feature_vector)

    def td_value_approximation(self, learning_rate=0.0005,epochs=100000,
                               fourier=True,ord = 5):
        self.state_value = self.policy_evaluation(self.policy)
        # 输入检查
        if (not isinstance(learning_rate, float) or
            not isinstance(epochs, int) or
            not isinstance(fourier, bool) or
            not isinstance(ord, int)):
            raise TypeError('learning rate and epochs should be float')
        if learning_rate < 0 or epochs < 0 or ord<0:
            raise ValueError('learning rate and epochs should be >0')
        episode_lenth = epochs
        start_state = np.random.randint(self.state_space_size)
        start_action = np.random.choice(range(self.action_space_size),
                                        p=self.mean_policy[start_state])
        # print(start_state, start_action)
        # print(self.mean_policy[start_state])
        episode = self.obtain_episode(self.policy, start_state, start_action, episode_lenth)
        # 采样
        dim = (ord+1)**2 if fourier else np.arange(ord+2).sum()
        # print(dim) # dim=36
        w = np.random.default_rng().normal(size = dim)
        # np.random.default_rng().normal
        # 从正态分布（也称为高斯分布）中生成随机样本的常用方法
        # print(w)
        rmse = []
        value_approximation = np.zeros(self.state_space_size) # shape =(25,1)
        for epoch in range(epochs):
            reward = episode[epoch]["reward"]
            state=  episode[epoch]["state"]
            next_state = episode[epoch]["next_state"]
            # r + γV'
            target = reward+self.gama*np.dot(self.gfv(fourier, next_state, ord), w)
            # error = = target - V
            error = target - np.dot(self.gfv(fourier, state, ord), w)
            # 梯度
            gradient =  self.gfv(fourier, state, ord)
            w = w +learning_rate * gradient * error
            for state in range(self.state_space_size):
                value_approximation[state] = np.dot(self.gfv(fourier, state, ord), w)
            rmse.append(np.sqrt(np.mean((value_approximation-self.state_value[state])**2)))
            print(epoch)
        # X, Y = np.meshgrid(np.arange(1, 6), np.arange(1, 6))
        # Z = self.state_value.reshape(5, 5)
        # Z1 = value_approximation.reshape(5, 5)
        # # 绘制 3D 曲面图
        # fig = plt.figure(figsize=(8, 6))  # 设置图形的尺寸，宽度为8，高度为6
        # ax = fig.add_subplot(121, projection='3d')
        # ax.plot_surface(X, Y, Z)
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('State Value')
        # z_min = -5
        # z_max = -2
        # ax.set_zlim(z_min, z_max)
        # ax1 = fig.add_subplot(122, projection='3d')
        # ax1.plot_surface(X, Y, Z1)
        # ax1.set_xlabel('X')
        # ax1.set_ylabel('Y')
        # ax1.set_zlabel('Value Approximation')
        # ax1.set_zlim(z_min, z_max)
        # fig_rmse = plt.figure(figsize=(8, 6))  # 设置图形的尺寸，宽度为8，高度为6
        # ax_rmse = fig_rmse.add_subplot(111)
        #
        # # 绘制 rmse 图像
        # ax_rmse.plot(rmse)
        # ax_rmse.set_title('RMSE')
        # ax_rmse.set_xlabel('Epoch')
        # ax_rmse.set_ylabel('RMSE')
        # plt.show()
        return value_approximation

    def gfv_a(self, fourier: bool, state: int, action: int, ord: int) -> np.ndarray:
        """
        get_feature_vector_with_action,这就是使用傅里叶特征函数或者多项式做基函数的过程
        :param fourier: 是否使用傅里叶特征函数
        :param state: 状态
        :param ord: 特征函数最高阶次数/傅里叶q(对应书)
        :return: 代入state后的计算结果,shape is (ord+1)**3,因为对三个变量（x,y,a)求了基函数
        """

        if state < 0 or state >= self.state_space_size or action < 0 or action >= self.action_space_size:
            raise ValueError("Invalid state/action value")
        feature_vector = []
        y, x = self.env.state2pos(state) + (1, 1)  # state 转换为位置形式

        if fourier:
            # 归一化到 -1 到 1
            x_normalized = x / self.env.size
            y_normalized = y / self.env.size
            action_normalized = action / self.action_space_size
            for i in range(ord + 1):
                for j in range(ord + 1):
                    for k in range(ord + 1):
                        feature_vector.append(np.cos(np.pi * (i * x_normalized + j * action_normalized + k * y_normalized))) # cos(pi (i+j+k))

        else:
            # 归一化到 0 到 1
            state_normalized = (state - (self.state_space_size - 1) * 0.5) / (self.state_space_size - 1)
            action_normalized = (action - (self.action_space_size - 1) * 0.5) / (self.action_space_size - 1)
            for i in range(ord + 1):
                for j in range(i + 1):
                    feature_vector.append(state_normalized ** (ord - i) * action_normalized ** j)
        return np.array(feature_vector)

    def sarsa_value_approximation(self,learning_rate = 0.005,
                    epsilon= 0.1,num_episodes =10,fourier =True,ord =5):
        """
        这里和sarsa的逻辑是一样的，只是把计算q-value的部分 函数化了,更新的是w参数，函数为grv_a(s,a)
        采用np.dot(a,b)进行计算q-value
        运行结果正确

        :param learning_rate:
        :param epsilon:
        :param num_episodes:
        :param fourier:
        :param ord:
        :return: qvalue ,not the  state value,it is different from the td_value_approximation。当然也可以求出state value，就是qvalue的expectation
        """
        dim = (ord+1)**3 if fourier else np.arange(ord +2).sum() # 因为
        w = np.random.default_rng().normal(size = dim)
        # print("w",w) # 36,1
        qvalue_approximation = np.zeros((self.state_space_size,self.action_space_size))
        reward_list =[]
        length_list =[]
        rmse =[]
        policy_rmse =[]
        policy  = self.mean_policy.copy()
        next_state = 0
        # episode = self.obtain_episode(self.mean_policy, 0,0,length =num_episodes)
        for episode in range(num_episodes):
            print(episode)
            done = False
            self.env.reset()
            total_reward = 0
            episode_length = 0
            while not done:
                state = next_state
                action = np.random.choice(range(self.action_space_size),
                                          p = policy[state])
                _,reward,done,_,_ = self.env.step(action)
                # 这是采样的过程
                episode_length += 1
                total_reward += reward
                next_state = self.env.pos2state(self.env.agent_location)
                next_action = np.random.choice(np.arange(self.action_space_size),
                                               p = policy[next_state])
                # 正常sarsa写法
                # target  = reward + self.gama * self.qvalue[next_state,next_action]
                # print(w.shape)
                matix_next = self.gfv_a(fourier, next_state, next_action, ord)
                # print(matix_next.shape)
                target = reward + self.gama * np.dot(matix_next, w)
                matix_state =self.gfv_a(fourier, state,action, ord)
                error = target  - np.dot(matix_state, w)
                gradient = self.gfv_a(fourier, state, action,ord)
                # 参数更新
                w = w +learning_rate * gradient * error

                qvalue_approximation[state,action] = np.dot(self.gfv_a(fourier, state, action, ord), w)
                # find the best action
                qvalue_star = qvalue_approximation[state].max()
                action_star = qvalue_approximation[state].tolist().index(qvalue_star)

                for a in range(self.action_space_size):
                    if a == action_star:
                        policy[state,a] = 1- (self.action_space_size - 1) / self.action_space_size * epsilon
                    else:
                        policy[state,a] = 1/ self.action_space_size * epsilon
            print("episode={},length={},reward={}".format(episode, episode_length, total_reward))
            # print(qvalue_approximation)
        self.policy = qvalue_approximation
        # return qvalue_approximation



if __name__ == '__main__':
    env = grid_env.GridEnv(size=5, target=[2, 3],
                           forbidden=[[2, 2], [2, 1], [1, 1], [3, 3], [1, 3], [1, 4]],
                           render_mode='')  # 初始化环境  size,target,forbidden的初始化
    solver = Solve(env)
    # env.render()  # 显示网格世界
    # episode = solver.obtain_episode(solver.mean_policy, 0,
    #                               0, length=50)
    # print(episode) #{'state': 0, 'action': 0, 'reward': -10, 'next_state': np.int64(0), 'next_action': np.int64(2)}
    solver.sarsa_value_approximation()
    solver.show_policy()
    env.render()