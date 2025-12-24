import random
import time

import matplotlib.pyplot as plt
import numpy as np

import gridenv.render as render
import gridenv.grid_env as genv

# no model 方法
# 是指智能体（Agent）在不知道环境规则（即状态转移概率和奖励函数）的情况下，
# 通过直接与环境进行大量的“试错”交互，从积累的经验中学习如何做出最优决策。
# 有模型说明知道环境的游戏规则，action有限的情况下，是Stochastic，action无限的情况下是Determined
# Determined (确定性的)：意味着“必然的”、“唯一的”。给定相同的输入，总是产生完全相同的结果。没有随机性，没有运气成分。
# Stochastic (随机性的)：意味着“概率性的”、“不确定的”。给定相同的输入，结果可能会不同，遵循某种概率分布。

"""
no model 需要采样，因为policy是不可知的
no model方法的分类主要在

"""


class Solve:
    def __init__(self,env=genv.GridEnv):
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
    def show_state_value(self,state_value,y_offset = 0.2):
        """
        展示  state value  图像展示
        :param state_value:
        :param y_offset:
        :return:
        """
        for state in range(self.state_space_size):
            # write_word(self,pos,word,
            # color: str = 'black',
            # y_offset=0,size_discount=1.0)
            self.env.render_.write_word(pos =self.env.state2pos(state),
                                word= str(round(state_value[state],1)),
                                y_offset=y_offset,
                                size_discount= 0.7
                                )

    def obtain_episode(self,policy,start_state,start_action,length):
        """
        模拟的采样的步骤
        :param policy: 一个episode
        :param start_state:
        :param start_action:
        :param length: episode 长度
        :return: 很多S,A,R,S',A'的序列 ，一个episode
        """
        self.env.agent_location = self.env.state2pos(start_state)
        # now location
        episode = []
        # a list for s and a
        next_action =  start_action
        next_state = start_state
        while length >0:
            length -= 1
            state = next_state
            action = next_action
            _,reward ,done ,_,_ =self.env.step(action) # 获得的immediately reward 以及是否到达target
            # env 方法中 agent_location 已给定
            # step方法给出的是
            # step return observation, reward, terminated, False, info
            next_state = self.env.pos2state(self.env.agent_location) # 调用的step函数自动更新location
            # 转换到state格式
            next_action = np.random.choice(np.arange(len(policy[next_state])),p=policy[next_state])
            # 随机从policy【 next state】中选择一个策略
            # 这是一个采样
            episode.append({"state":state, "action":action, "reward":reward, "next_state":next_state,
                            "next_acton":next_action})
        return episode
    def mc_basic(self,length =30,epochs = 10):
        """
        这是基本蒙特卡罗方法，需要计算完成全episode的q,就是一个序列的reward相加完成之后再进行计算。
        优点在于：获取的q是无偏的
        缺点在于：太慢了

        need：1.采样序列：这个序列看情况，因为蒙特卡罗方法只更新走过的state，所以需要全state每一个episode都做一遍
        2.初始策略随便选。无关的。对于走过的state，action会对这个策略进行更新。

        当然这个更新是从target逐步扩散开来的

        只更新了采样到的这些state 的策略，其他的一律不管
        需要一个探索策略
        :param length:一个episode的长度
        :param epochs: 重复次数
        :return: None  更新了policy
        """
        for epoch in range(epochs):
            for state in range(self.state_space_size):
                for action in range(self.action_space_size):
                    episode = self.obtain_episode(self.policy,state,action,length)
                    # 对每一个初始state，action都做了一遍
                    # print(episode)
                    # policy初始选择的是探索的随机策略
                    g =0
                    for step in range(len(episode)-1,-1,-1):
                        g = episode[step]["reward"] + g * self.gama
                        # 计算长期收益,此时，q是无偏的，是通过每个r直接计算出来的
                    self.qvalue[state][action] = g
                    qvalue_star = self.qvalue[state].max()
                    # 计算最优q,就是最优策略
                    action_star = self.qvalue[state].tolist().index(qvalue_star)
                    # 有q_value 的index就是  最优 的action
                    self.policy[state]  = np.zeros(shape  =self.action_space_size)
                    # re init policy[state]
                    self.policy[state][action_star] =1
            print(epoch)

    def sarsa(self,alpha =0.1,epsilon =0.1,num_episodes =80):
        """
        基础sarsa 算法，还有一些变种
        need : s ,a,r,s' ,a'
        还是只关注了经过的路程中的state,其他的不关注
        解决了蒙特卡洛方法速度太慢的问题，因为蒙方法需要计算全部r，才能获得当前的q
        所以Sarsa方法的优点就是快。
        :param alpha: 学习率
        :param epsilon: epsilon就是贪婪策略的系数，经验化设置为0.1是最好的
        :param num_episodes: 迭代次数
        :return:None
        """
        qvalue_list  = [self.qvalue,self.qvalue+1]
        while num_episodes>0:
            done = False
            self.env.reset() # 设置了起始点为[0,0]
            num_episodes -=1
            total_reward = 0
            episode_length = 0
            next_state = 0 # 如果不修改env的self.agent_location参数，这个数值就不能改
            while not done:
                state = next_state # 这里限定了起始点，只能保证从起始点开始的路径上的state是最优的，其他不管。这实际上和上面直接把episode全采样了是一样的。只不过是分步采样了。
                action = np.random.choice(np.arange(self.action_space_size),
                                          p = self.policy[state]) # 初始采取的是随机策略，属于探索性策略
                # s,a
                _,reward ,done,_,_ =self.env.step(action= action)
                # 走一步，按照现在的s,a 为基准，可以走一步，不需要policy
                episode_length+=1
                total_reward += reward # r
                next_state = self.env.pos2state(self.env.agent_location)
                # s'
                next_action = np.random.choice(np.arange(self.action_space_size),
                                               p = self.policy[next_state])
                # a'是下一步的action，随机生成的
                # 现在获取了所有的Sarsa五个参数
                target = reward + self.gama * self.qvalue[next_state,next_action] # q值初始化为全0
                # 这是贝尔曼公式  q [s]= r +gamma * q[s']
                error = self.qvalue[state,action] -target
                self.qvalue[state,action] -= error*alpha
                # TD learning 迭代
                qvalue_star = self.qvalue[state].max()
                action_star = self.qvalue[state].tolist().index(qvalue_star)
                # 这个state 的policy是计算完q后，立刻更新的
                # self.policy[state][action_star] = 1 # 这里不能直接设置为概率1。
                # 因为激进的结论需要激进的证据，按照现有的采样证据还不够。只能支持贪婪策略，保持一定的探索性余量
                # 要使用epsilon贪婪策略
                # 主要还是因为一个取样没有做充分
                for a in range(self.action_space_size):
                    if a == action_star:
                        # 这是现在采样的最优action
                        self.policy[state,a] = 1-(self.action_space_size-1)/self.action_space_size*epsilon
                    else:
                        self.policy[state,a] = epsilon/self.action_space_size

    def expected_sarsa(self,alpha = 0.1,epsilon = 1,num_episodes =80):
        """
        expected sarsa 就是计算q_value时，a’换成A‘，实际上算的就是V’
        need : s ,a,r,s' ,A'
        在计算sarsa时候，是直接取的a',这个a是基于策略采样的
        因为要对q'求expectation所以这个速度必然要比sarsa要慢一些的
        其他基本一致
        :param alpha:
        :param epsilon:
        :param num_episodes:
        :return:
        """
        init_num = num_episodes
        qvalue_list = [self.qvalue,self.qvalue+1]
        episode_index_list =[]
        reward_list =[]
        length_list = []
        while num_episodes>0:
            if epsilon > 0.1:
                epsilon -= 0.01
            episode_index_list.append(init_num - num_episodes)
            done = False
            self.env.reset()
            next_state =  0
            total_reward = 0
            episode_length = 0
            num_episodes -=1
            print(np.linalg.norm(qvalue_list[-1] - qvalue_list[-2],ord=1),num_episodes) # q
            while not done:
                state = next_state
                action = np.random.choice(np.arange(self.action_space_size),
                                        p = self.policy[state])
                _,reward ,done,_,_ =self.env.step(action= action)
                next_state =self.env.pos2state(self.env.agent_location)
                expect_qvalue = 0
                episode_length+=1
                # 记录
                total_reward += reward
                for next_action in range(self.action_space_size):
                    # 计算Q'
                    expect_qvalue += self.qvalue[next_state,next_action]*self.policy[next_state,next_action]
                target = reward + self.gama * expect_qvalue
                # target 通过贝尔曼公式计算得出
                error = self.qvalue[state,action] -target
                self.qvalue[state,action] -= error*alpha
                qvalue_star = self.qvalue[state].max()
                action_star = self.qvalue[state].tolist().index(qvalue_star)
                # 和sarsa一样
                for a in range(self.action_space_size):
                    # epsilon贪心
                    if a ==action_star:
                        self.policy[state,a] = 1-(self.action_space_size-1)/self.action_space_size*epsilon
                    else:
                        self.policy[state,a] = epsilon/self.action_space_size
            qvalue_list.append(self.qvalue.copy())
            reward_list.append(total_reward)
            length_list.append(episode_length)
        fig = plt.figure(figsize=(10, 10))
        self.env.render_.add_subplot_to_fig(fig=fig, x=episode_index_list, y=reward_list, subplot_position=211,
                                            xlabel='episode_index', ylabel='total_reward')
        self.env.render_.add_subplot_to_fig(fig=fig, x=episode_index_list, y=length_list, subplot_position=212,
                                            xlabel='episode_index', ylabel='total_length')
        fig.show()

    def nstep_sarsa(self,n=2,alpha =0.1,epsilon = 0.5,num_episodes =800):
        """
        还是感觉有点不对，算出来不是很好的策略。这代码写的非常反人类
        计算Q_value的时候，需要考虑到MC截断还是使用sarsa。因为采样n步可能在中途就到target了
        n-step sarsa 需要n+1步的s,a,r
        是sarasa和mc算法的结合，mc的基础上做了n步，再做sarsa
        :param n: n=1时就是sarsa,n=无穷，就是mc
        :param alpha:
        :param epsilon:
        :param num_episodes:
        :return:
        """
        length_list =[]
        qvalue_list = [self.qvalue]
        for episode in range(num_episodes):
            # 初始化
            self.env.reset() # 默认的起始点是0，0
            state = self.env.pos2state(self.env.agent_location)  # 这样每次的state都是(0，0)
            action = np.random.choice(np.arange(self.action_space_size),
                                      p= self.policy[state]) # random policy
            states = [state] # the list to save state
            actions  =[action] # save action
            rewards = [0.0] # 但是reward[0]是没有的

            t = 0
            T = float('inf')
            done = False
            while True:
                if not done:
                    # 直接调用obtain_episode，无法获得done参数
                    # 计算Q_value的时候，需要考虑到MC截断还是使用sarsa。因为采样n步可能在中途就到target了。
                    # 这个还是比sarsa要复杂一些的。需要done参数确定是否到达
                    obs,reward,done,_,_ = self.env.step(action= action)
                    next_state = self.env.pos2state(self.env.agent_location)
                    next_action = np.random.choice(np.arange(self.action_space_size),
                                                   p= self.policy[next_state])
                    states.append(next_state)
                    actions.append(next_action)
                    rewards.append(reward)
                    # 每次采样一步

                    if done:
                        T= t+1 # 实际的步数,终止
                tau = t -n + 1 # 现在的state
                print("tau",tau)
                # tau 表示start_state
                if tau >=0:
                    # G{tau:tau+n}
                    G  = 0.0
                    # n步之内的reward
                    for i in range(tau+1,min(tau+1,T)+1):
                        G+=(self.gama**(i -tau-1))*rewards[i] # 这样算不会出问题
                    # 未终止,+Q.终止了就不需要加了,算 Q   = R + gama* Q'
                    if tau + n<T:
                        s_n =states[tau+n] # 这是计算 q[tau+n]
                        a_n =actions[tau+n]
                        G+=(self.gama**n)* self.qvalue[s_n,a_n]
                    # 终止了就直接是MC截断了

                    s_tau = states[tau]
                    a_tau = actions[tau]

                    td_error = G - self.qvalue[s_tau,a_tau]
                    self.qvalue[s_tau,a_tau] += alpha * td_error

                    best_action = np.argmax(self.qvalue[state])
                    self.policy[state] = epsilon / self.action_space_size
                    self.policy[state, best_action] += 1 - epsilon
                    print("self.policy[state]",self.policy[state])
                if tau == T-1:
                    break

                t+=1
                if not done:
                    state = next_state
                    action = next_action

            # 每个 episode 结束后记录 Q 值变化

            qvalue_list.append(self.qvalue.copy())
            if len(qvalue_list) > 1:
                diff = np.linalg.norm(qvalue_list[-1] - qvalue_list[-2], ord=1)
                print(f"Episode {episode + 1}, L1 diff: {diff:.4f}")

    def nstep_sarsa2(self, n=2, alpha=0.1, epsilon=0.1, num_episodes=120):
        """
        这个代码写的有点问题，算出的策略不对，大致没有问题吧，算了
        n-step sarsa 需要n+1步的s,a,r
        是sarasa和mc算法的结合，mc的基础上做了n步，再做sarsa
        :param n: n=1时就是sarsa,n=无穷，就是mc
        :param alpha:
        :param epsilon:
        :param num_episodes:
        :return:
        """
        qvalue_list = [self.qvalue, self.qvalue + 1]
        while num_episodes > 0:
            # 迭代步数,这里是要求走完全程
            done = False
            self.env.reset()  # 每次都从0开始
            next_state = 0
            num_episodes -= 1
            episode_length = 0
            start_action = 0
            next_acton = 0
            while not done:
                # 起始点，要走完全程，不然更新不了，这是从target扩散出来的
                total_reward = 0 # 重置归零
                for step in range(n):
                    # 从起始点走n步.done必然Fasle
                    state = next_state
                    action = np.random.choice(np.arange(self.action_space_size),
                                              p=self.policy[state])
                    if step == 0:
                        # 存下起始的s,a用于计算q
                        start_action = action
                        start_state = state
                    # 走一步，env中的参数是跟着动的
                    _, reward, done, _, _ = self.env.step(action=action)
                    episode_length += 1
                    if not done:
                        next_state = self.env.pos2state(self.env.agent_location)
                        next_acton = np.random.choice(np.arange(self.action_space_size),
                                                      p=self.policy[next_state])
                        total_reward += reward * (self.gama ** step)
                    else:
                        # 到达target
                        print(state)
                        next_state = self.env.pos2state(self.env.agent_location)
                        next_acton = 4
                        # 4: np.array([0, 0])
                        # 这里属于人为干预了,不是太好
                        total_reward += reward * (self.gama ** step)
                        self.policy[state] =0
                        self.policy[state, 4] = 1
                    # R = r1 +r2*gamma+r3*gamma^2
                    # 计算immediate reward
                qvalue = total_reward + self.gama ** n * self.qvalue[next_state, next_acton]
                # 计算长期reward
                # 剩下的按照传统sarsa的做法
                error = self.qvalue[start_state, start_action] - qvalue  # 观测值的q_value  - target
                self.qvalue[start_state, start_action] -= error * alpha  # TD
                qvalue_star = self.qvalue[start_state].max()
                action_star = self.qvalue[start_state].tolist().index(qvalue_star)
                for a in range(self.action_space_size):
                    if a == action_star:
                        self.policy[start_state, a] = 1 - (
                                    self.action_space_size - 1) / self.action_space_size * epsilon
                    else:
                        self.policy[start_state, a] = epsilon / self.action_space_size
            qvalue_list.append(self.qvalue.copy())
            print(np.linalg.norm(qvalue_list[-1] - qvalue_list[-2], ord=1), num_episodes)

    def q_learning_onpolicy(self,alpha=0.001,epsilon=0.4,num_episodes=5000):
        """
        这里num_episodes 设置要大一些，否则无法收敛
        重点在于qvalue[s]=r+max(qvalue[s'])
        直接采用贝尔曼最优公式计算，有max就是贝尔曼最优公式
        可能需要的迭代次数要比之前的长一些
        由于始终是一个policy在迭代，所以是on policy的策略
        :param alpha:
        :param epsilon:
        :param num_episodes:
        :return:
        """
        init_num = num_episodes
        qvalue_list = [self.qvalue,self.qvalue+1]
        episode_index_list = []
        reward_list = []
        length_list = []
        while num_episodes > 0:
            episode_index_list.append(init_num-num_episodes)
            done = False
            self.env.reset()
            next_state = 0
            total_rewards = 0
            episode_length = 0
            num_episodes -= 1
            # 迭代自减
            print(np.linalg.norm(qvalue_list[-1] - qvalue_list[-2], ord=1))
            while not done:
                state = next_state
                action = np.random.choice(np.arange(self.action_space_size),
                                          p = self.policy[state])
                _,reward,done,_,_ = self.env.step(action=action)
                next_state = self.env.pos2state(self.env.agent_location)
                # 下一步位置
                episode_length += 1
                total_rewards += reward
                # immediate reward
                total_rewards +=reward
                next_qvalue_star = self.qvalue[next_state].max()
                target = reward  + next_qvalue_star * (1 - epsilon)*self.gama
                error =self.qvalue[state, action] - target
                # 观测 - 理想值
                self.qvalue[state, action] -= alpha * error
                qvalue_star = self.qvalue[state].max()
                action_star = self.qvalue[state].tolist().index(qvalue_star)
                for a in range(self.action_space_size):
                    if a == action_star:
                        self.policy[state, a] = 1 - (self.action_space_size - 1) / self.action_space_size * epsilon
                    else:
                        self.policy[state, a] = epsilon / self.action_space_size
            qvalue_list.append(self.qvalue.copy())
            reward_list.append(total_rewards)
            length_list.append(episode_length)
        fig = plt.figure(figsize=(10, 10))
        self.env.render_.add_subplot_to_fig(fig=fig, x=episode_index_list, y=reward_list, subplot_position=211,
                                            xlabel='episode_index', ylabel='total_reward')
        self.env.render_.add_subplot_to_fig(fig=fig, x=episode_index_list, y=length_list, subplot_position=212,
                                            xlabel='episode_index', ylabel='total_length')
        fig.show()

    def q_learning_offpolicy(self,alpha = 0.01,epsilon=0.4,episode_length = 2000):
        """
        off policy 的q_learning
        核心思想是把采样和策略更新分离，不使用一个策略
        那么函数obtain_episode就是一个很好的东西，只要一次产生很长的episode
        那么，采样就是随机探索；策略更新就和采样完全不相干了
        :param alpha:
        :param epsilon:
        :param episode_length:
        :return:
        """
        start_state = self.env.pos2state(self.env.agent_location)
        start_action = np.random.choice(np.arange(self.action_space_size),
                                        p= self.policy[start_state])
        episode =self.obtain_episode(self.mean_policy.copy(),start_state = start_state,
                                     start_action = start_action,length = episode_length)
        print("start_action",start_action)
        print("start_state",start_state)
        print("episode：",episode)
        for step in range(episode_length-1):
            # 留一个给next_state
            reward = episode[step]["reward"]
            state = episode[step]["state"]
            action = episode[step]["action"]
            next_state = episode[step+1]["state"]
            next_qvalue_star = self.qvalue[next_state].max()
            target =reward +self.gama* next_qvalue_star
            # 贝尔曼最优公式计算Q 的target值
            error = self.qvalue[state, action] - target
            self.qvalue[state, action] -= alpha * error
            action_star = self.qvalue[state].argmax()
            # 这里不使用epsilon 贪婪策略，直接取最优值。想要用也行。
            # 原因在于用贝尔曼最优公式得出的就是policy了。
            # 另一个原因是不需要现在的self.policy有什么探索性了
            self.policy[state] =0
            self.policy[state, action_star] = 1
            print("policy",self.policy)



if __name__ == '__main__':
    env = genv.GridEnv(size=5,
                    target=[2, 3],
    forbidden=[[2, 2], [2, 1], [1, 1], [3, 3], [1, 3], [1, 4]],
                    render_mode='')
    solver = Solve(env)
    solver.q_learning_offpolicy()
    solver.show_policy()
    solver.show_state_value(solver.state_value)
    env.render()

