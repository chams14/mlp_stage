import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def plot(x, y, z, title):
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.view_init(30, 135)
    ax.plot_surface(x, y, z, cmap='viridis')
    ax.set_xlabel('Speed v')
    ax.set_ylabel('Angle θ')
    ax.set_zlabel('T(v,θ)')
    ax.set_title(title)
    plt.show()


def t(v, a, g=9.81):
    return (v * v * np.sin(2 * a)) / g


#%%
# Exercise 2
x_val = np.linspace(0, 10, 1000)
y_val = np.linspace(0, np.pi / 2, 1000)  # pi/2 = 90°C radiant
X, Y = np.meshgrid(x_val, y_val)
Z = t(X, Y)
plot(X, Y, Z, 'Graph of the function T')

#%%
# Exercise 3
def simulation_euler(v, a, g=9.81, h=0.01):
    x, y = 0, 0
    xd1 = v * np.cos(a)
    yd1 = v * np.sin(a) - g * h
    while y >= 0:
        x += h * xd1
        y += h * yd1
        yd1 += - g * h
    return x


x_val = np.linspace(0, 10, 100)
y_val = np.linspace(0, np.pi / 2, 100)
X, Y = np.meshgrid(x_val, y_val)

Z = np.zeros_like(X)
for i, v in enumerate(x_val):
    for j, a in enumerate(y_val):
        Z[j, i] = simulation_euler(v, a)
plot(X, Y, Z, 'System solution by Euler')


#%%
# Exercise 4
def ball(y, time, g):
    return [y[2], y[3], 0, - g]  # xd1,yd1


def simulation_odeint(v, a, g=9.81):
    time = np.linspace(0, 10, 1000)
    xd1 = v * np.cos(a)
    yd1 = v * np.sin(a) - g * time[1]
    y0 = [0, 0, xd1, yd1]
    sol = odeint(ball, y0, time, args=(g, ))

    x = np.array(sol[:, 0])
    y = np.array(sol[:, 1])

    ind = np.argmax(y < 0)  # altitude : if y < 0 => cannon-ball on the floor
    return x[ind]  # horizontal : distance from the canon


x_val = np.linspace(0, 10, 100)
y_val = np.linspace(0, np.pi / 2, 100)
X, Y = np.meshgrid(x_val, y_val)

Z = np.zeros_like(X)
for i, v in enumerate(x_val):
    for j, a in enumerate(y_val):
        Z[j, i] = simulation_odeint(v, a)
plot(X, Y, Z, 'System solution by odeint')
