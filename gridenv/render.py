# 这是一个用于渲染的类，被grid_env.py 调用，只需要管理show，画图，写字的事情，不需要管理数据的存储和变化
import time
from typing import Union
import matplotlib.animation as animation
import matplotlib.patches  as patches
import matplotlib.pyplot as plt

import numpy as np
from matplotlib.patches import Patch
from pandas import factorize


class Render:
    def __init__(self,target,forbidden,size):
        """ Render 类的构造函数
            在grid_env.py  调用
        :param target:目标点的位置
        :param forbidden:障碍物区域位置
        :param size:网格世界的size 默认为 5x5
        """
        self.agent  =None
        self.target = target
        self.forbidden = forbidden
        self.size = size
        self.fig = plt.figure(figsize=(10,10),dpi=self.size*20)
        self.ax = plt.gca() # 可以通过这个对象 对坐标轴设置文字
        self.ax.xaxis.set_ticks_position('top')
        self.ax.invert_yaxis()
        self.ax.xaxis.set_ticks(range(0, size + 1))
        self.ax.yaxis.set_ticks(range(0, size + 1))
        self.ax.grid(True, linestyle="-", color="gray", linewidth="1", axis='both')
        self.ax.tick_params(bottom=False, left=False, right=False, top=False, labelbottom=False, labelleft=False,
                            labeltop=False)
        # 这些都是设置边框和图例的东西

        for y in range(size):
            self.write_word(pos= (-0.6,y),word=str(y+1),size_discount=0.8)
            self.write_word(pos= (y,-0.6), word=str(y + 1), size_discount=0.8)
        # 图例

        for pos in self.forbidden:
            self.fill_block(pos=pos)
        self.fill_block(pos= self.target,color ='darkturquoise')
        self.trajectory = []
        # trajectory 多个eposide
        # 画箭头，action
        self.agent = patches.Arrow(-10,-10,0.4,0,color='red',width=0.5)
        self.ax.add_patch(self.agent)

    def fill_block(self,pos: Union[list, tuple, np.ndarray], color: str = '#EDB120', width=1.0,
                   height=1.0) -> Patch:
        # 填充方块
        return self.ax.add_patch(
            patches.Rectangle((pos[0], pos[1]),
                              width=1.0,
                              height=1.0,
                              facecolor=color,
                              fill=True,
                              alpha=0.90,
                              ))

    def draw_randow_line(self,pos1,pos2):
        # a line between pos1 and pos2
        offset1 = np.random.uniform(low=-0.05,high=0.05,size =1)
        offset2 = np.random.uniform(low=-0.05, high=0.05, size=1)
        x = [pos1[0] + 0.5 , pos2[0]+0.5]
        y = [pos1[1] + 0.5, pos2[1] + 0.5]
        if pos1[0] == pos2[0]:
            x = [x[0]+offset1,x[1]+offset2]
        else:
            y = [y[0]+offset1,y[1]+offset2]
        self.ax.plot(x,y,color='g',scalex=False,scaley=False)

    def draw_circle(self,pos,radius,color,fill):
        # 在网格中心画圈
        return self.ax.add_patch(
            patches.Circle((pos[0]+0.5,pos[1]+0.5),
            radius = radius,
            facecolor= color,
            edgecolor= 'green',
            linewidth = 2,
            fill = fill
        ))

    def draw_action(self, pos: Union[list, tuple, np.ndarray], toward: Union[list, tuple, np.ndarray],
                    color: str = 'green', radius: float = 0.10) -> None:
        """
        将动作可视化
        :param radius: circle 的半径
        :param pos:网格的左下坐标
        :param toward:(a,b) a b 分别表示 箭头在x方向和y方向的分量 如果是一个 0 向量就画圆,这个就是从方块中点-》中点+toward偏移的箭头
        :param color: 箭头的颜色 默认为green
        :return:None
        """
        if not np.array_equal(np.array(toward), np.array([0, 0])):
            self.ax.add_patch(
                patches.Arrow(pos[0] + 0.5, pos[1] + 0.5, dx=toward[0],
                              dy=toward[1], color=color,
                              width=0.05 + 0.05 * np.linalg.norm(np.array(toward) / 0.5),
                              linewidth=0.5))
        else:
            self.draw_circle(pos= tuple(pos),color='white',radius=radius,fill=False)

    def write_word(self,pos,word,color: str = 'black',y_offset=0,size_discount=1.0):
        # 在pos位置写字
        self.ax.text(pos[0]+0.5,pos[1]+0.5,word,
                     size= size_discount*(30-2*self.size),ha= 'center',
                     va = 'center',color= color)
    def update_agent(self,pos,action,next_pos):
        # 更新agent的位置
        # trajectory is a list of [pos,action,nextpos]
        self.trajectory.append([tuple(pos),action,tuple(next_pos)])


    def show_frame(self,t = 0.5)->None:
        """
        显示figure 持续一段时间后 关闭
        :return: None
        """
        self.fig.show()
        plt.pause(t)

    def save_frame(self,name):
        # save frame
        self.fig.savefig(name+".jpg")

    def save_video(self,name):
        # save video
        anim = animation.FuncAnimation(
            self.fig,
            self.animate,init_func=self.init(),
            frame= len(self.trajectory),
            interval=25,repeat= False
        )
        anim.save(name+'.mp4')

        # init 和 animate 都是服务于animation.FuncAnimation
        # 具体用法参考matplotlib官网
    def init(self):
            pass

    def animate(self, i):
            print(i, len(self.trajectory))
            location = self.trajectory[i][0]
            action = self.trajectory[i][1]
            next_location = self.trajectory[i][2]
            next_location = np.clip(next_location, -0.4, self.size - 0.6)
            self.agent.remove()
            if action[0] + action[1] != 0:
                self.agent = patches.Arrow(x=location[0] + 0.5, y=location[1] + 0.5,
                                           dx=action[0] / 2, dy=action[1] / 2,
                                           color='b',
                                           width=0.5)
            else:
                self.agent = patches.Circle(xy=(location[0] + 0.5, location[1] + 0.5),
                                            radius=0.15, fill=True, color='b',
                                            )
            self.ax.add_patch(self.agent)

            self.draw_random_line(pos1=location, pos2=next_location)


    def draw_episode(self):
        for i in range(len(self.trajectory)):
            location= self.trajectory[i][0]
            next_location = self.trajectory[i][2]
            self.draw_randow_line(pos1=location,pos2=next_location)
            # 画出这个trajectory的路线轨迹


    def add_subplot_to_fig(self,fig,x,y,subplot_positon,xlabel,
                           ylabel,title=""):
        """
        在给定的位置上添加一个子图到当前的图中，并在子图中调用plot函数，设置x,y label和title。

        参数:
        x: 用于plot的x数据
        y: 用于plot的y数据
        subplot_position: 子图的位置，格式为 (row, column, index)
        xlabel: x轴的标签
        ylabel: y轴的标签
        title: 子图的标题
        """
        ax = fig.add_subplot(subplot_positon)
        # 调用plot函数绘制图形
        ax.plot(x, y)
        # 设置x,y label和title
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
if __name__ == '__main__':
    render = Render(target=[4,4],
                    forbidden=[np.array([1,2]),np.array([2,2])],
                    size=5)
    render.draw_action(pos=[4,3],toward=(0,0.4))

    for i in range(5):
        render.draw_randow_line(pos1=[0.5,1.5],pos2=[1.5,1.5])

    action_to_direction={
        0:np.array([-1,0]),
        1:np.array([0,1]),
        2: np.array([1, 0]),
        3: np.array([0, -1]),
        4: np.array([0, 0]),
    }

    uniform_policy = np.random.random(size=(25,5))

    for a in range(5):
        render.trajectory.append((a,a))

    render.show_frame(15)







