import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from tqdm import tqdm
from .ode_problem import ODEProblem


class NeuralODE:
    def __init__(self, neural_net: nn.Module, ode: ODEProblem):
        self.neural_net = neural_net.to(torch.float64)
        self.ode_problem = ode
        self.ode_t = None
        self.ode_data = None
        self.nn_data = None
        self.optimizer = optim.Adam(self.neural_net.parameters())

    def predict(self, t, x):
        combined = torch.cat(
            [torch.tensor([t]), torch.tensor(x)], dim=0)
        return self.neural_net(combined)

    def load_ode_data(self, t_arr, data_arr):
        ndim = self.ode_problem.u_0.shape[0]
        nsteps = len(self.ode_problem.t_steps)

        if t_arr.shape != (nsteps, ):
            raise Exception('Time array not in correct shape')

        if data_arr.shape != (ndim, nsteps):
            raise Exception('Data array not in correct shape')

        self.ode_t = t_arr
        self.ode_data = data_arr

    def ode_solve(self):
        if self.ode_data is None:
            self.ode_t, self.ode_data = self.ode_problem.solve()
        return self.ode_t, self.ode_data

    def nn_solve(self):
        if self.nn_data is None:
            sol = solve_ivp(fun=lambda t, x: self.predict(t, x).detach().numpy(),
                            t_span=self.ode_problem.t_span,
                            y0=self.ode_problem.u_0, t_eval=self.ode_t)
            if sol.status == 0:
                _, self.nn_data = sol.t, sol.y
                return self.ode_t, self.nn_data
            else:
                raise Exception('Neural Network ODE Solver failed to converge')

        else:
            return self.ode_t, self.nn_data

    def loss(self):
        if self.ode_data is None:
            raise Exception('Solve the ODE before training')

        # Get list of values from ode_data
        data = self.ode_data.tolist()
        data = list(map(list, zip(*data)))

        criterion = nn.MSELoss()

        loss_tensor = torch.tensor([0.0])
        for (t, x) in zip(self.ode_t, data):
            first_tensor = torch.tensor(
                self.ode_problem.f(t, torch.tensor(x))).double()
            second_tensor = self.predict(t, x)
            loss_tensor += criterion(first_tensor, second_tensor)
        return loss_tensor

    def train(self, num_epochs):
        for i in tqdm(range(num_epochs)):
            self.optimizer.zero_grad()
            self.loss().backward()
            self.optimizer.step()

            # Print the loss every 100 epochs
            if i % 100 == 0:
                print(f'Epoch {i}/{num_epochs}, Loss: {self.loss().item()}')

    def plot(self):
        if self.ode_data is None:
            raise Exception('Solve ODE before plotting')
        if self.nn_data is None:
            raise Exception('Solve neural network ODE before plotting')

        # Convert the arrays to numpy arrays for easier plotting
        t_np = np.array(self.ode_t)
        ode_data_np = np.array(self.ode_data)
        nn_data_np = np.array(self.nn_data)

        # Plot each line separately
        plt.figure(figsize=(10, 6))

        # Plot ODE solution
        for i in range(ode_data_np.shape[0]):
            plt.plot(t_np, ode_data_np[i, :],
                     label=f'ODE Solution {i + 1}', marker='o')

        # Plot Neural Network solution
        for i in range(nn_data_np.shape[0]):
            plt.plot(t_np, nn_data_np[i, :],
                     label=f'NN Solution {i + 1}', marker='x')

        # Add labels and a legend
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()

        # Show the plot
        plt.show()
