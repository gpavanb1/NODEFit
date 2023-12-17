import torch
import torch.nn as nn
import torch.optim as optim
from torchsde import sdeint
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# TODO: Add noise_type and sde_type as arguments


class SDE:
    def __init__(self, drift_nn, diffusion_nn, noise_type="diagonal", sde_type="ito", numerical_method="euler"):
        super().__init__()
        self.drift_nn = drift_nn
        self.diffusion_nn = diffusion_nn
        self.noise_type = noise_type
        self.sde_type = sde_type
        self.numerical_method = numerical_method
        self.drift_opt = optim.Adam(self.drift_nn.parameters())
        self.diffusion_opt = optim.Adam(self.diffusion_nn.parameters())

    def f(self, t, y):
        combined = torch.cat(
            (t * torch.ones((y.shape[0], 1), dtype=y.dtype), y), dim=1)
        return self.drift_nn(combined)

    def g(self, t, y):
        combined = torch.cat(
            (t * torch.ones((y.shape[0], 1), dtype=y.dtype), y), dim=1)
        return self.diffusion_nn(combined)


class NeuralSDE:
    def __init__(self, drift_nn: nn.Module, diffusion_nn: nn.Module, t, data, batch_size=2):
        self.sde = SDE(drift_nn, diffusion_nn)
        self.t = torch.tensor(t).double()
        self.data = torch.tensor(data).double()
        self.nn_data = None
        self.batch_size = batch_size

        (nsteps, _) = self.data.shape
        if len(self.t) != nsteps:
            raise Exception('Time array not in correct shape')

        self.y0 = self.data[0].clone().repeat(batch_size, 1)

    def loss(self):
        if self.data is None:
            raise Exception('Load the data before training')

        criterion = nn.MSELoss()

        self.nn_data = sdeint(self.sde, self.y0, self.t,
                              method=self.sde.numerical_method)
        repeated_data = self.data.unsqueeze(2).repeat(1, 1, self.batch_size)

        loss_tensor = criterion(repeated_data, self.nn_data)
        return loss_tensor

    def train(self, num_epochs, print_every=100):
        for i in tqdm(range(num_epochs)):
            self.sde.drift_opt.zero_grad()
            self.sde.diffusion_opt.zero_grad()
            self.loss().backward()
            self.sde.drift_opt.step()
            self.sde.diffusion_opt.step()

            # Print the loss every 100 epochs
            if i % print_every == 0:
                print(f'Epoch {i}/{num_epochs}, Loss: {self.loss().item()}')

    def extrapolate(self, tf, npts=20):
        tspan = np.linspace(self.t[-1], tf, npts)
        result = sdeint(
            self.sde, self.nn_data[-1].clone(), torch.tensor(tspan),
            method=self.sde.numerical_method)
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
            plt.plot(t_np, np.mean(nn_data_np[:, i, :], axis=1),
                     label=f'NN Solution {i + 1}', marker='x')

        # Plot extrapolated data
        if extra_data is not None:
            plt.gca().set_prop_cycle(None)
            for i in range(extra_data_np.shape[1]):
                plt.plot(extra_data['time'], np.mean(extra_data_np[:, i, :], axis=1),
                         label=f'Extrapolated NN Solution {i + 1}', marker='x', linestyle='dotted')

        # Add labels and a legend
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()

        # Show the plot
        plt.show()
