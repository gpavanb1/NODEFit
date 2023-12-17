import torch
import numpy as np
import torch.nn as nn
from nodefit.neural_ode import NeuralODE
from nodefit.neural_sde import NeuralSDE

# Use CUDA if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

###
# DEFINE NETWORKS
###

# Neural ODE parameters
ndim, drift_nhidden, diffusion_nhidden = 2, 10, 2
drift_nn = nn.Sequential(
    nn.Linear(ndim+1, drift_nhidden),
    nn.Sigmoid(),
    nn.Linear(drift_nhidden, ndim)
).double().to(device)

diffusion_nn = nn.Sequential(
    nn.Linear(ndim+1, diffusion_nhidden),
    nn.Sigmoid(),
    nn.Linear(diffusion_nhidden, ndim)
).double().to(device)

###
# PROVIDE DATA
###

t = np.linspace(0, 5, 10)
# Provide data as list of lists with starting condition
data = np.array([[1., 1.],
                 [1.52210594, 1.23757532],
                 [2.0570346, 1.37814989],
                 [2.47603815, 1.46040018],
                 [2.75026795, 1.50703724],
                 [2.91602961, 1.5343292],
                 [3.01170625, 1.5498438],
                 [3.06584853, 1.5585547],
                 [3.09827458, 1.56379774],
                 [3.11650095, 1.56674226]])

###
# FIT USING NEURALODE
###
print('Performing fit using Neural ODE...')

neural_ode = NeuralODE(drift_nn, t, data)
neural_ode.train(2000)

# # Extrapolate the training data
extra_data = neural_ode.extrapolate(10)
neural_ode.plot(extra_data)

###
# FIT USING NEURALSDE
###
print('Performing fit using Neural SDE...')

neural_sde = NeuralSDE(drift_nn, diffusion_nn, t, data)
neural_sde.train(1)

# # Extrapolate the training data
extra_data = neural_sde.extrapolate(10)
neural_sde.plot(extra_data)
