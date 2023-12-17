import torch
import torchsde
import numpy as np


class SDEProblem:
    def __init__(self, f, g, u_0, t_span, npts=10):
        self.f = f
        self.g = g
        self.u_0 = u_0
        self.t_span = t_span
        self.t_steps = np.linspace(t_span[0], t_span[1], npts)

    def solve(self):
        t_span = (self.t_span[0], self.t_span[1])
        solver = torchsde.sdeint.OverdampedEuler(
            self.f, self.g, self.u_0, t_span, dt=t_span[1] - t_span[0])
        solution = solver(None, self.t_steps)
        return self.t_steps, solution
