import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


from Controller.My_math.maths import build_B
from Controller.controller_cons import control_law 
from GVF_trajectory.GVF_Circle import gvf_circumference

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

if __name__ == '__main__':
    gvf = gvf_circumference([0,0], 80)
    N = 4
    s = np.array([100, 90, 0, 100, 200, 90, 10, -30])
    Z = [(0,1), (1,2), (2,3)]
    system = integrator(s, 15, gvf, N, Z, 20, 0.9)

    t_final = 200
    sol = system.run_simulation(0.1, t_final)

    fig, (ax1, ax2) = plt.subplots(2,1)
    x_cir, y_cir = gvf.gen_circumference_points(1000)
    ax1.plot(x_cir, y_cir)
    ax1.axis("equal")


    states = sol

    #ax1.plot(states[0,:], states[1,:], color="green")
    #ax1.plot(states[2,:], states[3,:], color="blue")
    #ax1.plot(states[4,:], states[5,:], color="red")
    #ax1.plot(states[6,:], states[7,:], color="brown")
    n = len(states[0,:])
    n=n-1
    ax1.scatter(states[0,n], states[1,n], color="green", marker="x")
    ax1.scatter(states[2,n], states[3,n], color="blue", marker="x")
    ax1.scatter(states[4,n], states[5,n], color="red", marker="x")
    ax1.scatter(states[6,n], states[7,n], color="brown", marker="x")

    B = build_B(Z, N)
    theta = np.zeros((N,len(states[0,:])))

    for i in range(N):
        theta[i, :] = np.atan2(states[2*i+1, :], states[2*i, :])


    e_theta = np.zeros((len(Z),len(states[0,:])))
    for i in range(len(Z)):
        index1, index2 = Z[i]
        c2 = np.cos(theta[index1, :])
        s2 = np.sin(theta[index1, :])
        c1 = np.cos(theta[index2, :])
        s1 = np.sin(theta[index2, :])
        e_theta[i, :] = np.atan2(s1 * c2 - c1 * s2, c1 * c2 + s1 * s2)

    t = np.linspace(0, t_final, len(states[0,:]))
    E_theta = B @ e_theta
    ax2.plot(t, E_theta[0, :])
    ax2.plot(t, E_theta[1, :])
    ax2.plot(t, E_theta[2, :])
    ax2.plot(t, E_theta[3, :])
    print(e_theta[2,:])
    plt.show()




