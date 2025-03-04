import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm

def t(v, a, g=9.81):
    return (v * v * np.sin(2 * a)) / g

def generate_data(nb, bruit=0.1):
    v = np.linspace(0, 10, nb)
    a = np.linspace(0, np.radians(90), nb)
    np.random.shuffle(v)
    z = t(v, a) + bruit * np.sin(a)
    return v, a, np.column_stack((v, a)), z.reshape(-1, 1)


v, a, x_data_train, y_data_train = generate_data(1000, 0.01)
x_train = torch.tensor(x_data_train, dtype=torch.float32)
y_train = torch.tensor(y_data_train, dtype=torch.float32)

v2, a2, x_data_train2, y_data_train2 = generate_data(1000, 0.1)
x_train2 = torch.tensor(x_data_train2, dtype=torch.float32)
y_train2 = torch.tensor(y_data_train2, dtype=torch.float32)

#%%
device = torch.device('cpu')
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Linear(2, 256)
        self.c2 = nn.Linear(256, 256)
        self.c3 = nn.Linear(256, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.c1(x))
        x = self.activation(self.c2(x))
        x = self.c3(x)
        return x


model = NeuralNetwork().to(device)
print(model)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train(x_data, y_data, model, criterion, optimizer):
    model.train()
    for batch in range(100):
        pred = model(x_data.to(device))
        loss = criterion(pred, y_data.to(device))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def test(x_data, y_data, model, criterion):
    model.eval()
    with torch.no_grad():
        pred = model(x_data.to(device))
        loss = criterion(pred, y_data.to(device)).item()
    return loss


#%%
loss = []
loss2 = []
for epochs in tqdm(range(1000)):
    current_loss = 0.0
    train(x_train, y_train, model, criterion, optimizer)
    loss.append(test(x_train, y_train, model, criterion))
    train(x_train2, y_train2, model, criterion, optimizer)
    loss2.append(test(x_train2, y_train2, model, criterion))
print("Training Done!")

#%%
x = np.arange(1000)
plt.figure()
plt.plot(x, loss, color='orange')
plt.plot(x, loss2, color='blue')
plt.legend(['train (0.01)', 'train (0.1)'])
plt.yscale('log')
plt.title("MLP PyTorch Loss")
plt.show()

#%%
model.eval()
with torch.no_grad():
    y_pred = model(x_train.to(device))

plt.figure()
ax = plt.axes(projection='3d')
ax.view_init(30, 135)
ax.scatter(v, a, y_train, color='blue', alpha=0.5)
ax.scatter(v, a, y_pred, color='red')
ax.legend(['y_train', 'y_pred'])
ax.set_xlabel('Speed v')
ax.set_ylabel('Angle θ')
ax.set_zlabel('T(v,θ)')
ax.set_title("MLP PyTorch")
plt.show()

#%%
V, A = np.meshgrid(np.sort(v), a)
z_grid = t(V, A) + 0.01 * np.sin(A)
Xx = np.column_stack((V.flatten(), A.flatten()))
X = torch.tensor(Xx, dtype=torch.float32)

model.eval()
with torch.no_grad():
    pred = model(X.to(device))
y_pred = pred.reshape(V.shape)

plt.figure()
ax = plt.axes(projection='3d')
ax.view_init(30, 135)
ax.plot_surface(V, A, z_grid, cmap='viridis')
ax.plot_wireframe(V, A, y_pred, color='black', linewidth=0.5, alpha=0.7)
ax.set_xlabel('Speed v')
ax.set_ylabel('Angle θ')
ax.set_zlabel('T(v,θ)')
ax.set_title("MLP PyTorch")
ax.legend(['y_train', 'y_pred'])
plt.show()