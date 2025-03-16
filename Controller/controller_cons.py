import numpy as np

def control_law(states, gvf, v, N, B, kb, kphi):

    #------------- Consnsus -------------
    L = B @ B.T
    theta = np.zeros((N,1))
    for i in range(N):
        theta[i] = np.atan2(states[i, 1], states[i, 0])

    u = -(L @ theta)
    u = (u + np.pi) % (2 * np.pi) - np.pi  
    u = u * kb 
    theta_dot = np.zeros((N, 1)) 
    r0 = gvf.radius
    for i in range(N):
        if u[i] < - r0:
            u[i] = -r0 * 0.9
        elif u[i] > 2 * r0:
            u[i] = r0 * 2

        theta_dot[i] = 1 / (r0 + u[i])
    u_dot =- kb * (L @ theta_dot)
    #------------- Behavior -------------
    gamma = u**2 + 2 * r0 * u
    gamma_dot = 2 * u_dot * (u + r0)

    #-------------   GVF    -------------
    J = np.zeros((2, N))
    phi = np.zeros((N, 1))
    for i in range(N):
        aux = gvf.grad_phi(states[i,:])
        J[0, i] = aux[0]
        J[1, i] = aux[1]
        phi[i] = gvf.phi(states[i,:])
    H = gvf.hess_phi()
    #-------------   f cal   -------------
    f = np.zeros((N,2))
    for i in range(N):
        orn_x = J[0, i] * (1 / (J[0, i]**2 + J[1, i]**2)) * (gamma_dot[i] + kphi * (gamma[i] - phi[i]))
        orn_y = J[1, i] * (1 / (J[0, i]**2 + J[1, i]**2)) * (gamma_dot[i] + kphi * (gamma[i] - phi[i]))
        orn_norm = np.sqrt(orn_x**2 + orn_y**2)

        if orn_norm > v:
            f[i,0] = v * orn_x / orn_norm
            f[i,1] = v * orn_y / orn_norm
        else:
            vt_x = J[1, i]
            vt_y = -J[0, i]
            vt_norm = np.sqrt(vt_x**2 + vt_y**2)
            vt_x = vt_x / vt_norm
            vt_y = vt_y / vt_norm
            alpha = np.sqrt(v**2 - orn_norm**2)
            f[i,0] = alpha * vt_x + orn_x
            f[i,1] = alpha * vt_y + orn_y

    return f