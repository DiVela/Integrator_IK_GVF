import numpy as np

from Controller.controller_cons import control_law 

class integrator():
    def __init__(self, init_state, init_speed, graf):
        self.state0 = init_state
        self.speed = init_speed
        self.B = graf

    def dynamics(self, t, state, N):
        states = state.reshape((N,2))
        states_dot = np.zeros((N,2))
        states_dot = control_law(states, N, self.B)
        return states_dot.flatten()
    



N = 3
s = np.array([[1,2, 2, 3, 3, 4]])
B = np.array([[1, 0],[-1, 1],[0, -1]])
system = integrator(s, 5, B)
system.dynamics(10, s, N)




