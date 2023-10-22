import matplotlib.pyplot as plt
import numpy as np

# input parameter
den = 2200                  # ガラス密度 [kg/m^3]
cp = 840                    # 比熱 [J/(kg K)]
cond = 1.1                  # 熱伝導率 [W/(m K)]
temp_bc = 263.15            # 境界温度
temp_init = 293.15          # 初期温度
lx = 3e-3                   # 窓の厚さ
nx = 16                     # 分割数
tend = 0.5                  # 計算終了時刻
dt = 1e-3                   # 時間幅
tout = 0.1                  # 出力時間間隔
alpha = cond / (den * cp)   # 温度伝導率
dx = lx / (nx - 1)          # 空間分割数
nt = int(tend / dt)         # ステップ数
nout = int(tout / dt)       # 出力数

# initial condition
temp = np.full(nx, temp_init)
time = 0.0
temp_new = np.zeros(nx)

# Boundary condition
temp[0] = temp_init         # Dirichlet @ x=0
temp[nx - 1] = temp_init    # Dirichlet @ x=Lx

# graph data array
ims = []
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
gx = np.zeros(nx)
for i in range(nx):
    gx[i] = i * dx

# Plot initial condition
im_line_init, = ax.plot(gx, temp, label='Initial Condition')
# time loop
for n in range(1, nt + 1):
    # FTCS
    for i in range(1, nx - 1):
        # 陽解法の離散化式
        temp_new[i] = temp[i] + alpha * dt / dx**2\
              * (temp[i + 1] - 2 * temp[i] + temp[i - 1])
    # update
    for i in range(1, nx - 1):
        temp[i] = temp_new[i]
    # Boundary condition
    temp[0] = temp_init         # Dirichlet @ x=0

    temp[nx - 1] = temp_bc      # Dirichlet @ x=Lx

    time += dt

    if n % nout == 0:
        print('n: {0:7d}, time: {1:8.1f},\
              temp: {2:10.6f}'.format(n, time, temp[nx - 1]))
        im_line, = ax.plot(gx, temp, label=f't = {time:.2f}s')
        ims.append(im_line)

ax.set_xlabel('x [m]')
ax.set_ylabel('Temperature [C]')
ax.grid()
ax.legend()
plt.show()
fig.savefig("aero_sim_5-1.png", dpi=300)
