import torch
import torch.nn as nn
import torch.optim as optim
from torchsde import sdeint
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from .constants import DEVICE


class SDE:
    def __init__(self, drift_nn, diffusion_nn, noise_type="diagonal", sde_type="ito", numerical_method="euler"):
        """
        Initializes a Stochastic Differential Equation (SDE) model.

        Parameters:
            - drift_nn (nn.Module): Neural network representing the drift term in the SDE.
            - diffusion_nn (nn.Module): Neural network representing the diffusion term in the SDE.
            - noise_type (str): Type of noise term. Default is "diagonal".
            - sde_type (str): Type of SDE. Default is "ito".
            - numerical_method (str): Numerical method for solving the SDE. Default is "euler". For Ito, euler, milstein and srk are available. For Stratanovich, midpoint, euler_heun, heun, milstein, and log_ode can be used
        """
        super().__init__()
        self.drift_nn = drift_nn
        self.diffusion_nn = diffusion_nn
        self.noise_type = noise_type
        self.sde_type = sde_type
        self.numerical_method = numerical_method
        self.drift_opt = optim.Adam(self.drift_nn.parameters())
        self.diffusion_opt = optim.Adam(self.diffusion_nn.parameters())

    def f(self, t, y):
        """
        Computes the drift term of the SDE at a given time and state.

        Parameters:
            - t (torch.Tensor): Time.
            - y (torch.Tensor): State.

        Returns:
            torch.Tensor: Drift term of the SDE.
        """
        combined = torch.cat(
            (t * torch.ones((y.shape[0], 1), dtype=y.dtype).to(DEVICE), y), dim=1).to(DEVICE)
        return self.drift_nn(combined)

    def g(self, t, y):
        """
        Computes the diffusion term of the SDE at a given time and state.

        Parameters:
            - t (torch.Tensor): Time.
            - y (torch.Tensor): State.

        Returns:
            torch.Tensor: Diffusion term of the SDE.
        """
        combined = torch.cat(
            (t * torch.ones((y.shape[0], 1), dtype=y.dtype).to(DEVICE), y), dim=1).to(DEVICE)
        return self.diffusion_nn(combined)


class NeuralSDE:
    def __init__(self, drift_nn: nn.Module, diffusion_nn: nn.Module, t, data, batch_size=2):
        """
        Initializes a Neural SDE model.

        Parameters:
            - drift_nn (nn.Module): Neural network representing the drift term in the SDE.
            - diffusion_nn (nn.Module): Neural network representing the diffusion term in the SDE.
            - t (numpy.ndarray or list): Time array.
            - data (numpy.ndarray or list): Time series data.
            - batch_size (int): Number of trajectories in each batch. Default is 2.
        """
        self.sde = SDE(drift_nn, diffusion_nn)
        self.t = torch.tensor(t).double().to(DEVICE)
        self.data = torch.tensor(data).double().to(DEVICE)
        self.nn_data = None
        self.batch_size = batch_size

        (nsteps, _) = self.data.shape
        if len(self.t) != nsteps:
            raise Exception('Time array not in correct shape')

        self.y0 = self.data[0].clone().repeat(batch_size, 1).to(DEVICE)

    def loss(self):
        """
        Computes the loss between the observed data and the Neural SDE predictions.

        Returns:
            torch.Tensor: Loss value.
        """
        if self.data is None:
            raise Exception('Load the data before training')

        criterion = nn.MSELoss()

        self.nn_data = sdeint(self.sde, self.y0, self.t,
                              method=self.sde.numerical_method).to(DEVICE)
        repeated_data = self.data.unsqueeze(2).repeat(
            1, 1, self.batch_size).to(DEVICE)

        loss_tensor = criterion(repeated_data, self.nn_data)
        return loss_tensor

    def train(self, num_epochs, print_every=100):
        """
        Trains the Neural SDE model using gradient descent.

        Parameters:
            - num_epochs (int): Number of training epochs.
            - print_every (int): Frequency of printing the loss during training. Default is 100.
        """
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
        """
        Extrapolates the Neural SDE solution beyond the observed time range.

        Parameters:
            - tf (float): Final time for extrapolation.
            - npts (int): Number of points for extrapolation. Default is 20.

        Returns:
            dict: Extrapolated time and values.
        """
        tspan = np.linspace(self.t[-1], tf, npts)
        result = sdeint(
            self.sde, self.nn_data[-1].clone().to(
                DEVICE), torch.tensor(tspan).to(DEVICE),
            method=self.sde.numerical_method).to(DEVICE)
        return {"time": tspan, "values": result}

    def plot(self, extra_data=None):
        """
        Plots the observed data, Neural Network solution, and extrapolated data (if provided).

        Args:
            extra_data (dict): Dictionary containing extrapolated time and corresponding values. Note that the plot is performed for the mean of all trajectories in the batch

        Returns:
            None
        """
        if self.data is None:
            raise Exception('Load data before plotting')
        if self.nn_data is None:
            raise Exception('Fit neural network before plotting')

        # Convert the arrays to numpy arrays for easier plotting
        t_np = self.t.cpu().numpy()
        data_np = self.data.cpu().numpy()
        nn_data_np = self.nn_data.detach().cpu().numpy()
        if extra_data is not None:
            extra_data_np = extra_data['values'].detach().cpu().numpy()

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
