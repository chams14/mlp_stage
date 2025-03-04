import numpy as np
import matplotlib.pyplot as plt
from sympy import *
import tensorflow as tf
from tqdm import tqdm


def plot_scatter(x, y, z, title):
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.view_init(30, 135)
    ax.scatter(x, y, z, s=20)
    ax.set_xlabel('Speed v')
    ax.set_ylabel('Angle θ')
    ax.set_zlabel('T(v,θ)')
    ax.set_title(title)
    plt.show()


def plot_scatter_rg(x, y, z, xr, yr, reg, title):
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.view_init(30, 135)
    ax.scatter(x, y, z)
    ax.plot_surface(xr, yr, reg, cmap='viridis')
    ax.set_xlabel('Speed v')
    ax.set_ylabel('Angle θ')
    ax.set_zlabel('T(v,θ)')
    ax.set_title(title)
    plt.show()


def plot_grad(x, y, dx, dgx, dgy, minx, miny, title):
    plt.figure()
    plt.plot(x, y, color='black')
    plt.scatter(dx, dgy, color='red')
    plt.scatter(minx, miny, color='green')
    plt.title(title)
    plt.show()

    plt.figure()
    plt.plot(dgx, dgy, color='black')
    plt.title(title)
    plt.show()


def t(v, a, g=9.81):
    return (v * v * np.sin(2 * a)) / g


# Exercise 1
def generate_data(nb, bruit=0.1):
    v = np.linspace(0, 10, nb)
    a = np.linspace(0, np.radians(90), nb)
    np.random.shuffle(v)
    z = t(v, a) + bruit * np.sin(a)
    plot_scatter(v, a, z, "Data")
    return np.column_stack((v, a)), z

x_data, y_data = generate_data(1000, 0.01)
print(x_data.shape)
print(y_data.shape)

#%%
# Exercise 2
def calculate_t(x, y, v, a):
    min_d = float('inf')
    dist = 0
    for i in range(len(x)):
        d = np.linalg.norm(np.array([v, a]) - (x[i][0], x[i][1]))
        if d < min_d:
            min_d = d
            dist = y[i]
    return dist


print("[Speed =", 10, ", angle =", 0.6, "]:", calculate_t(x_data, y_data, 10, 0.6))


#%%
# Exercise 5
def lin_reg_mp(dx, dy):
    x = np.array([[d[0], d[1], 1] for d in dx])
    mp_inv = np.linalg.pinv(x)
    c = mp_inv @ dy

    v = np.array([d[0] for d in dx])
    at = np.array([d[1] for d in dx])
    vr, atr = np.meshgrid(np.sort(v), np.sort(at))
    reg = c[0] * vr + c[1] * atr + c[2]
    plot_scatter_rg(v, at, dy, vr, atr, reg, 'Linear regression graph of Learning data with Moore-Penrose')


lin_reg_mp(x_data, y_data)


#%%
# Exercise 6
def reg_mp(dx, dy):
    x = np.array([[d[0],
                   d[0]**2,
                   d[0] * np.sin(d[1]),
                   d[0] * np.sin(2 * d[1]),
                   (d[0]**2) * np.sin(d[1]),
                   (d[0]**2) * np.sin(2 * d[1])]
                  for d in dx])
    mp_inv = np.linalg.pinv(x)
    c = mp_inv @ dy

    xv = np.array([d[0] for d in dx])
    ya = np.array([d[1] for d in dx])
    v, a = np.meshgrid(np.sort(xv), np.sort(ya))
    reg = c[0]*v + c[1]*(v**2) + c[2]*v*np.sin(a) + c[3]*v*np.sin(2*a) + c[4]*(v**2)*np.sin(a) + c[5]*(v**2)*np.sin(2*a)
    plot_scatter_rg(xv, ya, dy, v, a, reg, 'Regression graph of Learning data with Moore-Penrose')

reg_mp(x_data, y_data)

#%%
# Exercise 7
def critical_point(function):
    x = Symbol('x')
    # derivative
    df = diff(function, x)
    df2 = diff(df, x)
    # derivative = 0 -> critical points
    cp = solve(df, x)
    # second derivative -> nature of critical points
    local_max = 'inf'
    local_min = []
    for p in range(len(cp)):
        n = df2.subs(x, cp[p])
        if re(n.evalf()) > 0:
            print("Critical points -->", re(cp[p].evalf()), ": local minimum")
            local_min.append(re(cp[p].evalf()))
        elif re(n.evalf()) < 0:
            print("Critical points -->", re(cp[p].evalf()), ": local maximum")
            local_max = re(cp[p].evalf())
        else:
            print("Critical points -->", re(cp[p].evalf()), ": indeterminate nature")
    return local_min, local_max


def finite_difference(local_min, initial_condition, h=1e-5, delta=0.02):
    x = np.linspace(-2, 2, 1000)
    y = x ** 4 - 5 * x ** 2 + x + 10

    dg = initial_condition
    dgx = []
    dgy = []
    dx = []
    for i in range(70):
        grad = (((dg+h) ** 4 - 5 * (dg+h) ** 2 + (dg+h) + 10) - (dg ** 4 - 5 * dg ** 2 + dg + 10)) / h
        dg = dg - delta * grad
        dgx.append(i)
        dx.append(dg)
        dgy.append(dg ** 4 - 5 * dg ** 2 + dg + 10)
    min_y = [m ** 4 - 5 * m ** 2 + m + 10 for m in local_min]
    plot_grad(x, y, dx, dgx, dgy, local_min, min_y, "Gradient descent with FD : h = 1e-5, δ = 0.02")


def automatic_difference(local_min, initial_condition, delta=0.02):
    x = np.linspace(-2, 2, 1000)
    y = x ** 4 - 5 * x ** 2 + x + 10

    dg = tf.Variable(initial_condition, trainable=True)
    dgx = []
    dgy = []
    dx = []
    for i in range(70):
        with tf.GradientTape() as tape:
            predict_y = dg ** 4 - 5 * dg ** 2 + dg + 10
        grad = tape.gradient(predict_y, dg)
        dg.assign_sub(delta * grad)
        dgx.append(i)
        dx.append(dg.numpy())
        dgy.append(dg.numpy() ** 4 - 5 * dg.numpy() ** 2 + dg.numpy() + 10)
    min_y = [m ** 4 - 5 * m ** 2 + m + 10 for m in local_min]
    plot_grad(x, y, dx, dgx, dgy, local_min, min_y, "Gradient descent with Tensorflow : δ = 0.02")


f = Symbol('x') ** 4 - 5 * Symbol('x') ** 2 + Symbol('x') + 10
min_local, max_local = critical_point(f)
finite_difference(min_local, float(max_local))
automatic_difference(min_local, float(max_local))


#%%
# Exercise 8 - affine regression
def dg_regression_affine(dx, dy, epoch=5000, delta=0.001):
    a = tf.Variable([0.], dtype=tf.float32)
    b = tf.Variable([0.], dtype=tf.float32)
    c = tf.Variable([0.], dtype=tf.float32)

    xi = tf.constant(dy, dtype=tf.float32)

    def loss(a, b, c):
        return tf.reduce_mean(tf.square((a * dx[:, 0] + b * dx[:, 1] + c) - xi))

    dgx = []
    dgy = []
    with tf.device('/cpu:0'):
        for i in range(epoch):
            with tf.GradientTape() as tape:
                lo = loss(a, b, c)
            grad = tape.gradient(lo, [a, b, c])
            a.assign_sub(delta * grad[0])
            b.assign_sub(delta * grad[1])
            c.assign_sub(delta * grad[2])
            dgx.append(i)
            dgy.append(lo.numpy())

    v = np.array([d[0] for d in dx])
    at = np.array([d[1] for d in dx])
    vr, atr = np.meshgrid(np.sort(v), np.sort(at))
    reg = a * vr + b * atr + c
    plot_scatter_rg(v, at, dy, vr, atr, reg, 'Linear regression graph of Learning data by GD')

    plt.figure()
    plt.plot(dgx, dgy, color='black')
    plt.yscale('log')
    plt.title("GD for linear regression with Tensorflow : Niter=5000, δ = 0.001")
    plt.show()

dg_regression_affine(x_data, y_data)


#%%
# Exercise 8 - regression
def dg_regression(dx, dy, epoch=50000, delta=0.000005):
    a1 = tf.Variable([0.], dtype=tf.float32)
    a2 = tf.Variable([0.], dtype=tf.float32)
    a3 = tf.Variable([0.], dtype=tf.float32)
    a4 = tf.Variable([0.], dtype=tf.float32)
    a5 = tf.Variable([0.], dtype=tf.float32)
    a6 = tf.Variable([0.], dtype=tf.float32)

    xi = tf.constant(dy, dtype=tf.float32)

    def loss(a1, a2, a3, a4, a5, a6):
        v = dx[:, 0]
        a = dx[:, 1]
        r = a1*v + a2*(v**2) + a3*v*np.sin(a) + a4*v*np.sin(2*a) + a5*(v**2)*np.sin(a) + a6*(v**2)*np.sin(2*a)
        return tf.reduce_mean(tf.square(r - xi))

    # adaptive learning rate
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=delta)

    dgx = []
    dgy = []
    with tf.device('/cpu:0'):
        for i in tqdm(range(epoch)):
            with tf.GradientTape() as tape:
                lo = loss(a1, a2, a3, a4, a5, a6)
            grad = tape.gradient(lo, [a1, a2, a3, a4, a5, a6])
            optimizer.apply_gradients(zip(grad, [a1, a2, a3, a4, a5, a6]))
            dgx.append(i)
            dgy.append(lo.numpy())

    print(a1.numpy(), a2.numpy(), a3.numpy(), a4.numpy(), a5.numpy(), a6.numpy())

    xv = np.array([d[0] for d in dx])
    ya = np.array([d[1] for d in dx])
    v, a = np.meshgrid(np.sort(xv), np.sort(ya))
    reg = a1*v + a2*(v**2) + a3*v*np.sin(a) + a4*v*np.sin(2*a) + a5*(v**2)*np.sin(a) + a6*(v**2)*np.sin(2*a)
    plot_scatter_rg(xv, ya, dy, v, a, reg, 'Regression graph of Learning data with DG')

    plt.figure()
    plt.plot(dgx, dgy, color='black')
    plt.yscale('log')
    plt.title("GD for regression with Tensorflow : Niter=50000, initial δ = 0.000005")
    plt.show()

dg_regression(x_data, y_data)