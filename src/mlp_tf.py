import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def t(v, a, g=9.81):
    return (v * v * np.sin(2 * a)) / g

def generate_data(nb, bruit=0.1):
    v = np.linspace(0, 10, nb)
    a = np.linspace(0, np.radians(90), nb)
    np.random.shuffle(v)
    z = t(v, a) + bruit * np.sin(a)
    return v, a, np.column_stack((v, a)), z

v, a, x_train, y_train = generate_data(1000, 0.01)
v2, a2, x_train2, y_train2 = generate_data(1000, 0.1)

#%%
# Tensorflow-keras
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(256, input_shape=(2,), activation='sigmoid'))  # (2, 256)
model.add(tf.keras.layers.Dense(256, activation='sigmoid'))  # (256, 256)
model.add(tf.keras.layers.Dense(1, activation='linear'))  # (256, 1)
model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))
print(model.summary())

with tf.device('/GPU:0'):
    print("------Train data------")
    train = model.fit(x_train, y_train, epochs=1000, batch_size=100)
    print("\n------Train 2 data------")
    test = model.fit(x_train2, y_train2, epochs=1000, batch_size=100)

#%%
plt.plot(train.history['loss'], color='orange')
plt.plot(test.history['loss'], color='blue')
plt.legend(['train (0.01)', 'train (0.1)'])
plt.yscale('log')
plt.title("MLP Tensorflow Loss")
plt.show()

#%%
y_pred = model.predict(x_train)

plt.figure()
ax = plt.axes(projection='3d')
ax.view_init(30, 135)
ax.scatter(v, a, y_train, color='blue', alpha=0.5)
ax.scatter(v, a, y_pred, color='red')
ax.legend(['y_train', 'y_pred'])
ax.set_xlabel('Speed v')
ax.set_ylabel('Angle θ')
ax.set_zlabel('T(v,θ)')
ax.set_title("MLP Tensorflow")
plt.show()

#%%
V, A = np.meshgrid(np.sort(v), a)
z_grid = t(V, A) + 0.01 * np.sin(A)
X = np.column_stack((V.flatten(), A.flatten()))

pred = model.predict(X)
y_pred = pred.reshape(V.shape)

plt.figure()
ax = plt.axes(projection='3d')
ax.view_init(30, 135)
ax.plot_surface(V, A, z_grid, cmap='viridis')
ax.plot_wireframe(V, A, y_pred, color='black', linewidth=0.5, alpha=0.7)
ax.set_xlabel('Speed v')
ax.set_ylabel('Angle θ')
ax.set_zlabel('T(v,θ)')
ax.set_title("MLP Tensorflow")
ax.legend(['y_train', 'y_pred'])
plt.show()