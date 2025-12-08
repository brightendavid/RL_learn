import random
import time

import matplotlib.pyplot as plt
import numpy as np

import gridenv.render as render
import gridenv.grid_env as genv
"""
使用Policy Iteration和Value Iteration需要满足条件：
* Action对State的影响和回报 P(State', Reward | State, Action)是已知的
* 就是说是有model
* 网格世界是有model env,但是实际上很多env都是无model

此外，这两种方法都是全局的优化，一定是所有state都有到达target的最优路径的
在手撕算法的情况下，都是画Q-table解决。
Policy Iteration 要更快一些

 策略迭代 (Policy Iteration)
 是有初始策略的
初始化：随机选择一个策略作为初始值，比如说不管什么状态，一律朝下走，即P(Action = 朝下走 | State) = 1，P(Action = 其他Action | State) = 0
第一步 策略评估 (Policy Evaluation)：根据当前的策略计算V(s),内部迭代，求V（s）
第二步 策略提升 (Policy Improvement)：计算当前状态的最好Action，更新策略， 
不停的重复策略评估和策略提升，直到策略不再变化为止.

价值迭代 (Value Iteration)
初始化：所有state的价值V(s) = 0
第一轮迭代：对于每个state，逐一尝试上、下、左、右四个Action
记录Action带来的Reward、以及新状态 V(s')
选择最优的Action，更新V(s) = Reward + V
第二轮迭代：
同上
第三轮迭代：
...
直到所有的S中 都找到了最优的action,
停止

"""

class Solve:
    def __init__(self,env=genv.GridEnv):
        self.gama = 0.9
        self.env = env
        self.action_space_size = env.action_space_size
        self.state_space_size = env.size ** 2
        self.policy = self.random_greed_policy() # 策略初始化

        self.reward_space_size, self.reward_list = len(self.env.reward_list), self.env.reward_list
        self.state_value = np.zeros(shape=self.state_space_size)
        self.qvalue = np.zeros(shape=(self.state_space_size, self.action_space_size))
        self.mean_policy = np.ones(shape=(self.state_space_size, self.action_space_size)) / self.action_space_size
        # self.writer = SummaryWriter("logs")  # 实例化SummaryWriter对象


    def random_greed_policy(self):
        """
        一个随机的贪心策略
        就是every state has a policy action,pi=1
        :return: policy
        """
        # shape is state_size * action_size
        policy = np.zeros(shape=(self.state_space_size,self.action_space_size))
        for s_index in range(self.state_space_size):
            action = np.random.choice(range(self.action_space_size))
            policy[s_index,action]=1
        return policy
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
    def policy_evaluation(self,policy,tolerance=0.001,step=10):
        """
        策略评估，自循环的迭代，迭代出来最后的V
        这个是贝尔曼公式
        V = R + γPV
        结果是V = V'  就行了
        不更新policy，迭代V获取v pi.用于衡量当前策略，不是求解最优策略。
        求解贝尔曼公式，获得state value最优化V pi,使用的是迭代法
        满足 step经过了一定步骤，或者V-V'误差小于tolerance
        # bootstrapping
        自己生成自己的方法，自举式
        :param policy:
        :param tolerance:
        :param step:
        :return: State_value_k  是贝尔曼最优
        """
        state_value_k= np.ones(self.state_space_size) # 贝尔曼最优Vπ
        state_value = np.zeros(self.state_space_size)
        while np.linalg.norm(state_value_k-state_value,ord=1)>tolerance:
            # V，Vπ 的一范式小于tolerance,迭代求解
            state_value = state_value_k.copy()
            for state in range(self.state_space_size):
                value = 0
                for action in range(self.action_space_size):
                    value += policy[state,action] *self.calculate_qvalue(state_value=state_value_k.copy(), state=state, action=action)
                    # value = state_value,，就是V= PQ,Q是V中单方向的一个采样
                    #
                    # 这里再把Q按照action概率相加就是V
                state_value_k[state] = value
        return state_value_k

    def calculate_qvalue(self,state,action,state_value):
        """
        用在
        policy_improvement，policy_evaluation
        两个需要求Q-table的场合
        在计算之前先计算state value
        依据贝尔曼公式
        V= R +γPV 进行计算
        Q=  R+γPQ

        print(self.Psa.shape)
         # (25, 5, 25)   Psa, 就是 s->s'   的概率，  s经过5个 action  后到达  s'   s,s'有25种情况，就是网格的大小
        print(self.Rsa.shape)
        # (25, 5, 4)   Psa   s经过5个action 获得的  的reword
        计算qvalue elementwise形式，就是immediate reward+长期reward
        :param state: 对应的state
        :param action: 对应的action
        :param state_value: 状态值
        :return: 计算出的结果
        """
        qvalue =0
        # 这是immediately reward
        for i in range(self.reward_space_size):
            qvalue +=(self.reward_list[i] * self.env.Rsa[state,action,i])
        # 这是长期reward,通过V‘计算
        for next_state in range(self.state_space_size):
            #  γ P V
            qvalue += (self.gama *
                       self.env.Psa[state,action,next_state] *
                       state_value[next_state])
        return qvalue

    def value_iteration(self,tolerance=0.001,step  =100):
        """
        可以通过迭代V，求出最优策略
        价值迭代 (Value Iteration)important
            初始化：所有state的价值V(s) = 0
            第一轮迭代：对于每个state，逐一尝试上、下、左、右四个Action
            记录Action带来的Reward、以及新状态 V(s')
            选择最优的Action，更新V(s) = Reward + V
            第二轮迭代：
            同上
            第三轮迭代：
            ...
            直到所有的S中 都找到了最优的action,
            停止
        :param tolerance:
        :param step:
        :return:
        """
        state_value_k = np.ones(self.state_space_size)
        # 初始化state_value 为全0
        while ((np.linalg.norm(state_value_k - self.state_value,ord=1))>tolerance
               and step >0):
            # 满足迭代条件
            step -=1
            self.state_value= state_value_k.copy()
            self.policy,state_value_k = self.policy_improvement(state_value_k.copy())
        return step
    def policy_iteration(self,tolerance=0.001,step =1000):
        """
         策略迭代 (Policy Iteration)
             是有初始策略的
            初始化：随机选择一个策略作为初始值，比如说不管什么状态，一律朝下走，即P(Action = 朝下走 | State) = 1，P(Action = 其他Action | State) = 0
            第一步 策略评估 (Policy Evaluation)：根据当前的策略计算V(s),内部迭代，求V（s）
            第二步 策略提升 (Policy Improvement)：计算当前状态的最好Action，更新策略，
            不停的重复策略评估和策略提升，直到策略不再变化为止.
        :param tolerance:
        :param step:
        :return: step剩余
        """
        # 需要一个初始化的policy,这个策略可以不好
        policy = self.random_greed_policy()
        while np.linalg.norm(policy - self.policy) > tolerance and step >0 :
            step -=1
            policy = self.policy.copy()
            # 迭代V，直到V = V’
            # 这个内部是自循环的
            self.state_value = self.policy_evaluation(policy,tolerance=tolerance,step=step)
            # 这个是对每个S，求最大action的动作
            self.policy,_ = self.policy_improvement(self.state_value)
        return step



    def policy_improvement(self,state_value):
        """
        是普通 policy_improvement 的变种 相当于是值迭代算法 也可以 供策略迭代使用 做策略迭代时不需要 接收第二个返回值
        更新 qvalue ；qvalue[state,action]=reward+value[next_state]
        找到 state 处的 action*：action* = arg max(qvalue[state,action]) 即最优action即最大qvalue对应的action
        更新 policy ：将 action*的概率设为1 其他action的概率设为0 这是一个greedy policy

        就是做了一个找最大action的工作，并返回
        :param: state_value: policy对应的state value
        :return: improved policy, 以及迭代下一步的state_value
        """
        policy = np.zeros(shape=(self.state_space_size,self.action_space_size))
        state_value_k = state_value.copy()
        for state in range(self.state_space_size):
            # 遍历所有 的state
            q_value_list =[]
            for action in range(self.action_space_size):
                # 遍历所有的action
                q_value_list.append(self.calculate_qvalue(state,action,state_value.copy()))
                # 计算当前s 所有 action 的q值，就是手撕算法中的求q-table的过程
            # 手撕算法中的取q-table中每个s的最大action
            state_value_k[state] = max(q_value_list)
            # 设置这个action 的概率为1，其他都是0
            action_star = q_value_list.index(max(q_value_list))
            policy[state,action_star] = 1
        return policy, state_value_k



if __name__ == '__main__':
    env = genv.GridEnv(size=5,
                    target=[2, 3],
    forbidden=[[2, 2], [2, 1], [1, 1], [3, 3], [1, 3], [1, 4]],
                    render_mode='')
    solver = Solve(env)
    solver.policy_iteration()
    solver.show_state_value(solver.state_value)
    solver.show_policy()
    env.render()

