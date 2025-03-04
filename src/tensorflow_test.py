import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def f(x,y):
    return x*np.cos(y)

n = 25
xmin, xmax, ymin, ymax = -4.0, 4.0, -4.0, 4.0

VX = np.linspace(xmin, xmax, n)
VY = np.linspace(ymin, ymax, n)
X, Y = np.meshgrid(VX, VY)
Z = f(X, Y)

entree = np.append(X.reshape(-1,1), Y.reshape(-1,1), axis=1)
sortie = Z.reshape(-1,1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(10, input_dim=2, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.SGD(lr=0.01))
print(model.summary())

#%%
for k in range(1000):
    loss = model.train_on_batch(entree, sortie)
    print("Erreur :", loss)

sortie_produite = model.predict(entree)
ZZ = sortie_produite.reshape(Z.shape)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, color='blue', alpha=0.7)
ax.plot_surface(X, Y, ZZ, color='red', alpha=0.7)
plt.show()