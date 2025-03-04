import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# data 1
good = np.array([[0.1, 0.1], [0.1, 0.5], [0.3, 0.4], [0.4, 0.2], [0.6, 0.9]])
bad = np.array([[0.4, 0.4], [0.5, 0.6], [0.6, 0.3], [0.7, 0.6], [0.9, 0.2]])
# data 2
#good = np.array([[0.3, 0.4], [0.4, 0.2], [0.6, 0.9]])
#bad = np.array([[0.1, 0.5], [0.2, 0.2], [0.4, 0.4], [0.5, 0.6], [0.6, 0.3], [0.7, 0.6], [0.9, 0.2]])

plt.figure()
plt.scatter(good[:, 0], good[:, 1], c='green', marker='o')
plt.scatter(bad[:, 0], bad[:, 1], c='blue', marker='^')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.title("data")
plt.show()

data = np.vstack((good, bad))
# data 1
y = np.array([[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]])
# data 2
#y = np.array([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]])

#%%
def initialisation():
    # layer 2 -> W = 2 x 2      b = R^2
    w2 = np.random.randn(2, 2)
    b2 = np.random.randn(2, 1)
    # layer 3 -> W = 3 x 2      b = R^3
    w3 = np.random.randn(3, 2)
    b3 = np.random.randn(3, 1)
    # layer 4 -> W = 2 x 3      b = R^2
    w4 = np.random.randn(2, 3)
    b4 = np.random.randn(2, 1)
    return w2, b2, w3, b3, w4, b4

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def forward(x, w2, b2, w3, b3, w4, b4):
    a2 = sigmoid(w2 @ x + b2)
    a3 = sigmoid(w3 @ a2 + b3)
    a4 = sigmoid(w4 @ a3 + b4)
    return a2, a3, a4

def loss(w2, b2, w3, b3, w4, b4, x, y):
    f = sigmoid(w4 @ sigmoid(w3 @ sigmoid(w2 @ x + b2) + b3) + b4)
    return 1/10 * np.sum(1/2 * np.linalg.norm(y - f, ord=2))

def d(a):
    return a * (1 - a)

def backward(x, y, a2, a3, a4, w3, w4):
    db4 = d(a4) * (a4 - y)
    dw4 = db4 @ a3.T
    db3 = d(a3) * w4.T @ db4
    dw3 = db3 @ a2.T
    db2 = d(a2) * w3.T @ db3
    dw2 = db2 @ x.T
    return dw2, dw3, dw4, db2, db3, db4

def update(dw2, dw3, dw4, db2, db3, db4, w2, b2, w3, b3, w4, b4, lr=0.1):
    w4 = w4 - lr * dw4
    b4 = b4 - lr * db4
    w3 = w3 - lr * dw3
    b3 = b3 - lr * db3
    w2 = w2 - lr * dw2
    b2 = b2 - lr * db2
    return w2, b2, w3, b3, w4, b4

def predict(x, w2, b2, w3, b3, w4, b4):
  _, _, a4 = forward(x, w2, b2, w3, b3, w4, b4)
  return a4

def train(x, y, lr=0.1, epochs=100000):
    # initialisation w, b
    w2, b2, w3, b3, w4, b4 = initialisation()
    a2 = np.random.randn(2, 1)
    a3 = np.random.randn(3, 1)
    a4 = np.random.randn(2, 1)
    train_loss = []
    for _ in tqdm(range(epochs)):
        for m in range(0, 10):
            xx = np.array([[x[m][0]], [x[m][1]]])
            yy = np.array([[y[m][0]], [y[m][1]]])
            a2, a3, a4 = forward(xx, w2, b2, w3, b3, w4, b4)
            dw2, dw3, dw4, db2, db3, db4 = backward(xx, yy, a2, a3, a4, w3, w4)
            w2, b2, w3, b3, w4, b4 = update(dw2, dw3, dw4, db2, db3, db4, w2, b2, w3, b3, w4, b4, lr)
        epoch_loss = 0
        for m in range(0, 10):
            xx = np.array([[x[m][0]], [x[m][1]]])
            yy = np.array([[y[m][0]], [y[m][1]]])
            epoch_loss += loss(w2, b2, w3, b3, w4, b4, xx, yy)
        train_loss.append(epoch_loss)

    plt.figure()
    plt.plot(train_loss, label='training loss')
    plt.yscale('log')
    plt.title("Loss Evolution")
    plt.ylabel('Loss')
    plt.xlabel('Iterations')
    plt.show()
    return w2, b2, w3, b3, w4, b4, a2, a3, a4


w2, b2, w3, b3, w4, b4, a2, a3, a4 = train(data, y)
#%%
x = np.linspace(0, 1, 1000)
y = np.linspace(0, 1, 1000)
X, Y = np.meshgrid(x, y)
Z = np.zeros(X.shape)

for i in tqdm(range(X.shape[0])):
    for j in range(X.shape[1]):
        p = np.array([[X[i, j]], [Y[i, j]]])
        f = predict(p, w2, b2, w3, b3, w4, b4)
        Z[i, j] = 1 if f[0][0] > f[1][0] else 0

plt.figure()
plt.contourf(X, Y, Z, levels=[-0.5, 0.5, 1.5], colors=['grey', 'pink'], alpha=0.5)
plt.contour(X, Y, Z, levels=[0.5], colors='black')
plt.scatter(good[:, 0], good[:, 1], c='green', marker='o')
plt.scatter(bad[:, 0], bad[:, 1], c='blue', marker='^')
plt.legend(['good', 'bad'])
plt.title('Data and decision boundary')
plt.show()