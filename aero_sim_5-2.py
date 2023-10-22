"""
    @title  1次元定常熱伝導問題の数値解法（陽解法）
    @author Naito Masaki
    @date   2023/10/22
"""


import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata


class Window:
    def __init__(self, **kwargs):
        self.density = kwargs["density"]
        self.c_p = kwargs["c_p"]
        self.conductivity = kwargs["conductivity"]
        self.alpha = self.conductivity / (self.density * self.c_p)

        self.temp_1 = kwargs["temp_1"]
        self.temp_2 = kwargs["temp_2"]

        self.thickness = kwargs["thickness"]


def simulation(window, **kwargs):
    DIV_X = kwargs["div_x"]
    t_end = kwargs["t_end"]
    dt = kwargs["dt"]
    dx = window.thickness / (DIV_X - 1)
    DIV_TIME = int(t_end / dt)

    # 初期条件
    time = 0.0
    temp = np.full(DIV_X + 1, window.temp_1)
    temp_new = np.zeros(DIV_X + 1)

    # 境界条件
    temp[0] = window.temp_1
    temp[DIV_X] = window.temp_2

    # グラフ描画用配列
    time_buf = []
    pos_buf = []
    temp_buf = []

    # 初期条件をグラフ描画用リストに保存
    for i in range(0, DIV_X):
        time_buf.append(0)
        pos_buf.append(i * dx)
        temp_buf.append(temp[i])

    for n in range(1, DIV_TIME + 1):
        # 陽解法の離散化式
        for i in range(1, DIV_X - 1):
            temp_new[i] = temp[i] + window.alpha * dt / dx**2\
                * (temp[i + 1] - 2 * temp[i] + temp[i - 1])

        for i in range(1, DIV_X - 1):
            temp[i] = temp_new[i]

        temp[0] = window.temp_1
        temp[DIV_X - 1] = window.temp_2

        time = n * dt

        for i in range(0, DIV_X):
            time_buf.append(time)
            pos_buf.append(i * dx)
            temp_buf.append(temp[i])

    return time_buf, pos_buf, temp_buf


def plot_graph_3d(time, pos, temp):
    x, y = np.meshgrid(np.unique(time), np.unique(pos))
    z = griddata((time, pos), temp, (x, y))
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    ax.plot_surface(x, y, z, cmap="jet")
    ax.plot_wireframe(x, y, z, color="black", linewidth=0.3)
    ax.contour(x, y, z, colors="black")

    ax.set_xlabel('time [s]')
    ax.set_ylabel('position [m]')
    ax.set_zlabel("temperature [K]")

    ax.view_init(elev=30, azim=45)
    fig.savefig("aero_sim_5-2.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    window_cond = {
        "density": 2200,        # ガラス密度 [kg/m^3]
        "c_p": 840,             # 比熱 [J/(kg K)]
        "conductivity": 1.1,    # 熱伝導率 [W/(m K)]
        "temp_1": 293.15,       # 境界温度 [K]
        "temp_2": 263.15,       # 境界温度 [K]
        "thickness": 3e-3       # ガラス厚さ [m]
    }

    analysis_cond = {
        "div_x": 16,            # 空間分割数
        "t_end": 0.5,           # 解析終了時間 [s]
        "dt": 1e-3              # 時間分解能 [s]
    }

    glass_window = Window(**window_cond)

    time, pos, temp = simulation(glass_window, **analysis_cond)
    plot_graph_3d(time, pos, temp)
