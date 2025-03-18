import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from Controller.controller_cons import control_law 
from GVF_trajectory.GVF_Circle import gvf_circumference
from My_math.maths import build_B 

class integrator():
    def __init__(self, init_state, init_speed, gvf_traj, numUAV, graf, Kb, Kphi):
        self.state0 = init_state
        self.speed = init_speed
        self.B = graf
        self.gvf = gvf_traj
        self.N = numUAV
        self.kb = Kb
        self.kphi = Kphi 

    def dynamics(self, state):
        states = state.reshape((self.N,2))
        states_dot = np.zeros((self.N,2))
        states_dot = control_law(states, self.gvf, self.speed, self.N, self.B, self.kb, self.kphi)
        return states_dot.flatten()
    
    def run_simulation(self, dt, t_final):
        n = np.arange(0, t_final // dt)
        sol = np.zeros((len(self.state0), len(n)))
        
        sol[:,0] = self.state0
        for i in range(0,len(n)-1):
            sol[:,i+1] = self.dynamics(sol[:,i]) * dt + sol[:,i]
        return sol
    
    def print_solution_1N(self,sol, t, ax):
        states = sol.sol(t)

        ax.plot(states[0,:], states[1,:], color="green", marker='.')
        ax.plot(states[2,:], states[3,:], color="blue", marker='.')




