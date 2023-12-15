import numpy as np
from scipy.integrate import solve_ivp


class ODEProblem:
    def __init__(self, f, u_0, t_span, npts=10):
        self.f = f
        self.u_0 = u_0
        self.t_span = t_span
        self.t_steps = np.linspace(t_span[0], t_span[1], npts)

    def solve(self):
        sol = solve_ivp(fun=self.f, t_span=self.t_span, y0=self.u_0,
                        t_eval=self.t_steps)
        if sol.status == 0:
            return sol.t, sol.y
        else:
            raise Exception('ODE Solver failed to converge')
