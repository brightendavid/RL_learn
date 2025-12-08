import gymnasium as gym

# 创建环境实例
env = gym.make("GridWorld-v0", render_mode="human", size=5)

# 运行随机策略
observation, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()  # 随机动作
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        observation, info = env.reset()

env.close()