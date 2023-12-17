import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from tqdm import tqdm


class NeuralODE:
    def __init__(self, neural_net: nn.Module, t, data):
        self.neural_net = neural_net
        self.t = torch.tensor(t).double()
        self.data = torch.tensor(data).double()
        self.nn_data = None
        self.optimizer = optim.Adam(self.neural_net.parameters())

        (nsteps, _) = self.data.shape
        if len(self.t) != nsteps:
            raise Exception('Time array not in correct shape')

        self.y0 = self.data[0].clone()

    def predict(self, t, y):
        combined = torch.cat(
            [torch.tensor([t]), y.clone()], dim=0)
        return self.neural_net(combined)

    def loss(self):
        if self.data is None:
            raise Exception('Solve the ODE before training')

        criterion = nn.MSELoss()

        self.nn_data = odeint(self.predict, self.y0, self.t)

        loss_tensor = criterion(self.data, self.nn_data)
        return loss_tensor

    def train(self, num_epochs, print_every=100):
        for i in tqdm(range(num_epochs)):
            self.optimizer.zero_grad()
            self.loss().backward()
            self.optimizer.step()

            # Print the loss every 100 epochs
            if i % print_every == 0:
                print(f'Epoch {i}/{num_epochs}, Loss: {self.loss().item()}')

    def extrapolate(self, tf, npts=20):
        tspan = np.linspace(self.t[-1], tf, npts)
        result = odeint(
            self.predict, self.nn_data[-1].clone(), torch.tensor(tspan))
        return {"time": tspan, "values": result}

    def plot(self, extra_data=None):
        if self.data is None:
            raise Exception('Load data before plotting')
        if self.nn_data is None:
            raise Exception('Fit neural network before plotting')

        # Convert the arrays to numpy arrays for easier plotting
        t_np = self.t.numpy()
        data_np = self.data.numpy()
        nn_data_np = self.nn_data.detach().numpy()
        if extra_data is not None:
            extra_data_np = extra_data['values'].detach().numpy()

        # Plot each line separately
        plt.figure(figsize=(10, 6))

        # Plot time series
        for i in range(data_np.shape[1]):
            plt.plot(t_np, data_np[:, i],
                     label=f'Trained Data {i + 1}', marker='o')

        # Plot Neural Network solution
        plt.gca().set_prop_cycle(None)
        for i in range(nn_data_np.shape[1]):
            plt.plot(t_np, nn_data_np[:, i],
                     label=f'NN Solution {i + 1}', marker='x')

        # Plot extrapolated data
        if extra_data is not None:
            plt.gca().set_prop_cycle(None)
            for i in range(extra_data_np.shape[1]):
                plt.plot(extra_data['time'], extra_data_np[:, i],
                         label=f'Extrapolated NN Solution {i + 1}', marker='x', linestyle='dotted')

        # Add labels and a legend
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()

        # Show the plot
        plt.show()
