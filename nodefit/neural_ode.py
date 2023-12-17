import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from .constants import DEVICE


class NeuralODE:
    def __init__(self, neural_net: nn.Module, t, data):
        """
        A class representing a neural ordinary differential equation (Neural ODE) model to fit time-series data.

        Args:
            neural_net (nn.Module): The neural network model.
            t (array-like): Time values for the data.
            data (array-like): Observed data to be fitted by the Neural ODE.

        Attributes:
            neural_net (nn.Module): The neural network model.
            t (torch.Tensor): Time values for the data.
            data (torch.Tensor): Observed data to be fitted by the Neural ODE.
            nn_data (torch.Tensor): Neural network-generated data after fitting.
            optimizer (torch.optim.Optimizer): The optimizer used for training the neural network.
            y0 (torch.Tensor): Initial state for solving the ODE.

        Note:
            This class assumes that the provided neural network (`neural_net`) has a compatible architecture.
        """
        self.neural_net = neural_net
        self.t = torch.tensor(t).double().to(DEVICE)
        self.data = torch.tensor(data).double().to(DEVICE)
        self.nn_data = None
        self.optimizer = optim.Adam(self.neural_net.parameters())

        (nsteps, _) = self.data.shape
        if len(self.t) != nsteps:
            raise Exception('Time array not in correct shape')

        self.y0 = self.data[0].clone().to(DEVICE)

    def predict(self, t, y):
        """
        Predicts the next state using the neural network.

        Args:
            t (float): The current time.
            y (torch.Tensor): The current state.

        Returns:
            torch.Tensor: The predicted next state.
        """
        combined = torch.cat(
            [torch.tensor([t]).to(DEVICE), y.clone().to(DEVICE)], dim=0).to(DEVICE)
        return self.neural_net(combined)

    def loss(self):
        """
        Computes the mean squared error loss between observed and predicted data.

        Returns:
            torch.Tensor: The loss value.
        """
        if self.data is None:
            raise Exception('Load the data before training')

        criterion = nn.MSELoss()

        self.nn_data = odeint(self.predict, self.y0, self.t).to(DEVICE)

        loss_tensor = criterion(self.data, self.nn_data)
        return loss_tensor

    def train(self, num_epochs, print_every=100):
        """
        Trains the neural network to fit the observed data.

        Args:
            num_epochs (int): The number of training epochs.
            print_every (int): Print loss every `print_every` epochs.

        Returns:
            None
        """
        for i in tqdm(range(num_epochs)):
            self.optimizer.zero_grad()
            self.loss().backward()
            self.optimizer.step()

            # Print the loss every 100 epochs
            if i % print_every == 0:
                print(f'Epoch {i}/{num_epochs}, Loss: {self.loss().item()}')

    def extrapolate(self, tf, npts=20):
        """
        Extrapolates the solution of the Neural ODE to future time points.

        Args:
            tf (float): The final time for extrapolation.
            npts (int): Number of points for extrapolation.

        Returns:
            dict: Dictionary containing extrapolated times and corresponding values.
        """
        tspan = np.linspace(self.t[-1].cpu().item(), tf, npts)
        result = odeint(
            self.predict, self.nn_data[-1].clone().to(DEVICE), torch.tensor(tspan).to(DEVICE)).to(DEVICE)
        return {"time": tspan, "values": result}

    def plot(self, extra_data=None):
        """
        Plots the observed data, Neural Network solution, and extrapolated data (if provided).

        Args:
            extra_data (dict): Dictionary containing extrapolated time and corresponding values.

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
