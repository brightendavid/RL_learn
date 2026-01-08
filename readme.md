# 强化学习🐳
参考
https://github.com/jwk1rose/RL_Learning.git

* 做了一些注释
* 把model based单独拎出来
* 写代码的思想我认为非常有意思，所以说需要多参考别人的代码，这是非常有好处的

## 结构🔥

- [x] Something is Not TODO.

| 代码文件       |          功能           | 依赖 |
|:-----------|:---------------------:|---:|
| gridenv    |  这是定义网格世界环境，包含图形化部分   | ？？ |
| nomodel.py |      这是无模型的一些算法       | ？？ |
| solver.py  |      这是有模型的一些算法       | ？？ |
VFA.py| 这是使用函数化方法去估计qvalue的方法 | ？？ |
net_based.py|        这是dqn算法        | ？？ |

## Problems🍳
* nstep sarsa实现有问题
* PGM的采样无法实现足够的探索性，导致最后的策略有问题——这个问题可以修改reward之间的比值解决。
* ACM的最终策略在forbidden区域稍微有一些问题

# 思考

* 可以通过修改target,forbidden的reward使得ACM,PGM取得不同的效果，这是非常重要的。
* 一般结论，这些reward，等比修改一般是无效的。需要修改它们的比值。一般而言，target的reward要更大一些，否则就容易出现原地打转的情况，这个次有策略比瞎跑到forbidden区域里面扣分要好一些。


# 贝尔曼BE

$$
V(s)=\sum \pi(a|s)[\sum_{r}p(r|s,a)r+\gamma \sum_{s'} p(s'|s,a)v(s')]
$$

$$
q(s,a)=\sum_{r}p(r|s,a)r+\gamma \sum_{s'} p(s'|s,a)v(s')
$$

BOE就是加一个max

## ACM核心公式

> [!IMPORTANT]
>
> 请注意，强化学习的一般假设条件，就是搭梯子及其严重。包括贝尔曼公式中，假设了**马尔科夫性，一个状态只有上一个状态决定，是一个马尔科夫决策过程**。这一点是要非常注意的。当然搭梯子是可行的，只要最后的结果有用，而且假设不要太离谱就可以用。还有**奖励假设**，得到最多的score就是最优的策略。**i.i.d**假设，采样是独立同分布的。**延迟奖励**假设，非贪心，可能某一步是差的，但是总体是好的，取决于折扣因子的大小，这个因子作为一个超参数是试出来的。
>
> ## 具体而言
>
> 强化学习假设我们生活在一个**马尔可夫世界**里（忘记过去，只看现在），我们的目标是**赚取最多的积分**（奖励假设），为了达到这个目标，我们必须在这个世界里不断**尝试和犯错**（交互与试错），并在这个过程中学会**权衡眼前利益和长远利益**（折扣因子与延迟奖励）

因为Critic就是评论家网络评估策略的效果使用q-value去实现
$$
loss_{Critic} = r+\gamma q(s')-q(s)
$$

$$
w_{t+1}=w_{t}+\alpha[r+\gamma q(s')-q(s)]\bigtriangledown _{w}q(s)
$$

效果是给定一个state，输出一组q(s,a)，包含所有action的向量 

Actor网络生成策略
$$
loss_{Actor} = ln \pi(a|s)*q(s,a)
$$

$$
\theta _{t+1}=\theta _{t}+a\bigtriangledown ln \pi(a|s)*q(s,a)
$$

效果是给定一个state，输出一组π(s,a),包含这个state采取所有action的概率。

这里ln π(a|s)有一个推导过程的。

基础架构叫做QAC，加入一个baseline，就是actor网络中q变成q-v，就是a2c就是优势函数-AC。

## 感谢国科大刘俊明老师的机器学习课程👀

* 创造性得提出了教授强化学习
* 一些思路方法我是十分认可的
* 可以看出是想要教一些真东西的
* 惹出什么事端不要把师傅说出去就行了

## 计划👀
* 补全一些原作者没写的方法
* 加些注释，否则别人看不懂
* 使用现在主流的方法完成
* 可能在展示方面可以做一些不足


## 想法👀
* 我认为无论有没有完成强化学习课程的学习，都可以通过实现/借鉴这个仓库熟悉/预习强化学习
* 我认为现阶段这个仓库的注释写的还是非常完善的
* 一些基本理论，也尽量写上去
* 尽可能的说人话
* 我认为光听理论，不写代码。给你10年也学不会强化学习。这也对很多事情通用。
* 两条腿走路。一面是理论，一面是实践。
* 计算机专业应该会写代码，鼓掌👏👏👏。
* 我感觉强化学习做这些东西还是太基础了。可以尝试做一些游戏AI，自动脚本自动决策。
* 我看到一个很好玩的东西 https://github.com/wty-yy/KataCR.git  这个工作做得比我的本科毕设更有趣。工作量也是非常足够的。用视觉手段实现游戏AI皇室战争。
* https://github.com/Ronchy2000/Multi-agent-RL.git 我看到一个实现的非常完善的仓库，认为可以学习。里面还是有不少东西的

## 感谢西湖大学赵老师的视频课程❄️

<iframe src="//player.bilibili.com/player.html?isOutside=true&aid=388100433&bvid=BV1sd4y167NS&cid=998866354&p=1" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"></iframe>

* 这个课讲的非常不错，需要结合代码实践加深理解。

* 听明白需要一定的基础



## Star History

<a href="https://www.star-history.com/#brightendavid/RL_learn">

 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=brightendavid/RL_learn&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=brightendavid/RL_learn&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=brightendavid/RL_learn&type=Date" />
 </picture>
</a>

---

<p align="center">
  <em> ❤️ 感谢您的关注!</em><br><br>
  <img src="https://visitor-badge.laobi.icu/badge?page_id=brightendavid/RL_learn&style=for-the-badge&color=00d4ff" alt="Views">
</p>