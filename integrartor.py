import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from Controller.controller_cons import control_law 
from GVF_trajectory.GVF_Circle import gvf_circumference

class integrator():
    def __init__(self, init_state, init_speed, gvf_traj, numUAV, graf):
        self.state0 = init_state
        self.speed = init_speed
        self.B = graf
        self.gvf = gvf_traj
        self.N = numUAV

    def dynamics(self, t, state):
        states = state.reshape((self.N,2))
        states_dot = np.zeros((self.N,2))
        states_dot = control_law(states, self.gvf, self.speed, self.N, self.B, 20, 0.9)
        return states_dot.flatten()
    
    def run_simulation(self, t_final):
        sol = solve_ivp(
            self.dynamics, 
            [0, t_final],
            self.state0,
            method="RK45",
            dense_output=True,
            max_step=0.1
        )
        return sol
    
    def print_solution_2N(self,sol, t, ax):
        states = sol.sol(t)
        for i in range(states.shape[1]):
            ax.scatter(states[0,i], states[1,i], color="green", marker='.')
            ax.scatter(states[2,i], states[3,i], color="blue", marker='.')
            ax.scatter(states[4,i], states[5,i], color="red", marker='.')
            ax.scatter(states[6,i], states[7,i], color="yellow", marker='.')
            plt.pause(0.01)





            

        ax.scatter(states[0,states.shape[1]-1], states[1,states.shape[1]-1], color="green", marker='x', s=50)
        ax.scatter(states[2,states.shape[1]-1], states[3,states.shape[1]-1], color="blue", marker='x', s=50)
        ax.scatter(states[4,states.shape[1]-1], states[5,states.shape[1]-1], color="red", marker='x', s=50)
    

gvf = gvf_circumference([0,0], 50)

N = 4
s = np.array([100, 0, 100, 100, 0, 100, -100, 100])
B = np.array([[1, 0, 0], [-1 ,1, 0], [0, -1 , 1], [0, 0, -1]])
system = integrator(s, 15, gvf, N, B)
#system.dynamics(1, s)


t_final = 30
sol = system.run_simulation(t_final)

fig, ax = plt.subplots(1,1)
x, y = gvf.gen_circumference_points(10000)
ax.plot(x,y, color="black")
ax.axis("equal")
t = np.linspace(0, t_final, 100)
system.print_solution_2N(sol, t, ax)
plt.show()





