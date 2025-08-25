import numpy as np
from numpy.linalg import det, inv

from Controller.My_math.maths import build_B

def control_law(states, gvf, v, N, Z, kb, kphi, zd):
    #------------- Consnsus -------------
    r0 = gvf.radius
    B = build_B(Z, N)

    theta = np.zeros((N,1))
    for i in range(N):
        theta[i] = np.atan2(states[i, 1], states[i, 0])
    
    e_theta = np.zeros((len(Z),1))
    e_speed = np.zeros((len(Z),1))
    for i in range(len(Z)):
        index1, index2 = Z[i]
        c2 = np.cos(theta[index1])
        s2 = np.sin(theta[index1])
        c1 = np.cos(theta[index2])
        s1 = np.sin(theta[index2])
        v1 = v[index1]
        v2 = v[index2]  
        e_theta[i] = np.atan2(s1 * c2 - c1 * s2, c1 * c2 + s1 * s2) + zd[i] * np.pi/180
        e_speed[i] = v2 - v1

    kb = kb
    L = B.T @ B
    u = kb * (B @ (e_theta + inv(L) @ e_speed * 1/(kb * r0)))
    #u = kb * (B @ e_theta) + (B @ e_speed) * 1 / (r0)
    theta_dot = np.zeros((N, 1)) 
    
    for i in range(N):
        if u[i]<=-v[i]/r0 + v[i]/150:
            u[i] = -v[i]/(r0) + v[i]/150
        if u[i] >= v[i]/80 - v[i]/r0:
            u[i] = v[i]/80 - v[i]/r0

    for i in range(N):
        theta_dot[i] = v[i] / r0 + u[i]
    e_theta_dot = np.zeros((len(Z),1))

    for i in range(len(Z)):
        index1, index2 = Z[i]
        e_theta_dot[i] = theta_dot[index2] - theta_dot[index1]

    u_dot = kb * (B @ e_theta_dot)

    a = np.zeros((N,1))
    a_dot = np.zeros((N,1))
    for i in range(N):
        a[i] =  -u[i] * r0**2 / (v[i] + u[i] * r0)
        a_dot[i] = -r0**2 * u_dot[i] * v[i]/ (v[i] + u[i] * r0)**2
    #------------- Behavior -------------
    gamma = a**2 + 2 * r0 * a
    gamma_dot = 2 * a_dot * (a + r0) 
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

        if orn_norm > v[i]:
            f[i,0] = v[i] * orn_x / orn_norm
            f[i,1] = v[i] * orn_y / orn_norm
        else:
            vt_x = -J[1, i]
            vt_y = J[0, i]
            vt_norm = np.sqrt(vt_x**2 + vt_y**2)
            vt_x = vt_x / vt_norm
            vt_y = vt_y / vt_norm
            alpha = np.sqrt(v[i]**2 - orn_norm**2)
            f[i,0] = alpha * vt_x + orn_x
            f[i,1] = alpha * vt_y + orn_y

    return f

def control_law_elipse(states, gvf, v, N, Z, kb, kphi, kx, ky):
    #------------- Consnsus -------------
    B = build_B(Z, N)

    theta = np.zeros((N,1))
    for i in range(N):
        theta[i] = np.atan2(states[i, 1], states[i, 0])
    
    e_theta = np.zeros((len(Z),1))
    e_speed = np.zeros((len(Z),1))
    for i in range(len(Z)):
        index1, index2 = Z[i]
        c2 = np.cos(theta[index2])
        s2 = np.sin(theta[index2])
        c1 = np.cos(theta[index1])
        s1 = np.sin(theta[index1])
        v1 = v[index1]
        v2 = v[index2]
        e_theta[i] = np.atan2(s1 * c2 - c1 * s2, c1 * c2 + s1 * s2)
        e_speed[i] = v2 - v1

    #kb = kb / (np.pi * (N)) * max([gvf.a, gvf.b]) 
    u = - kb * (B @ e_theta) - kx * (B @ B.T @ B @ e_speed)
    
    for i in range(N):
        if u[i] <= 80 - min([gvf.a, gvf.b]):
            u[i] = 80 - min([gvf.a, gvf.b])
        elif (u[i] > 80 - min([gvf.a, gvf.b])) and (u[i] < 200 - max([gvf.a, gvf.b])):
            u[i] = u[i]
        elif u[i] >= 200 - max([gvf.a, gvf.b]):
            u[i] = 200 - max([gvf.a, gvf.b])
    
    theta_dot = np.zeros((N, 1)) 
    for i in range(N):
        theta_dot[i] = v[i] * (gvf.b + gvf.a + 2 * u[i])/((gvf.a + u[i]) * (gvf.b + u[i])) / 2

    e_theta_dot = np.zeros((len(Z),1))
    for i in range(len(Z)):
        index1, index2 = Z[i]
        e_theta_dot[i] = theta_dot[index1] - theta_dot[index2]

    u_dot = - kb * (B @ e_theta_dot)

    a = np.zeros((N,1))
    b = np.zeros((N,1))
    a_dot = np.zeros((N,1))   
    b_dot = np.zeros((N,1))
    gamma = np.zeros((N,1))
    gamma_dot = np.zeros((N,1))
    x_dot = np.zeros((N,1))
    y_dot = np.zeros((N,1))
    for i in range(N):
        a[i] = 1 / gvf.a**2 - 1/(gvf.a + u[i])**2
        b[i] = 1 / gvf.b**2 - 1/(gvf.b + u[i])**2
        a_dot[i] = 2 * u_dot[i] / (gvf.a + u[i])**3
        b_dot[i] = 2 * u_dot[i] / (gvf.b + u[i])**3
        x_dot[i] = v[i] * np.cos(theta[i])
        y_dot[i] = v[i] * np.sin(theta[i])
    #------------- Behavior -------------
        gamma[i] = (states[i,0] - gvf.x0)**2 * a[i] + (states[i,1] - gvf.y0)**2 * b[i]
        gamma_dot[i] = 2 * (states[i,0] - gvf.x0) * a[i] * x_dot[i] + 2 * (states[i,1] - gvf.y0) * b[i] * y_dot[i]
        gamma_dot[i] += a_dot[i] * (states[i,0] - gvf.x0)**2 + b_dot[i] * (states[i,1] - gvf.y0)**2
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

        if orn_norm > v[i]:
            f[i,0] = v[i] * orn_x / orn_norm
            f[i,1] = v[i] * orn_y / orn_norm
        else:
            vt_x = J[1, i]
            vt_y = -J[0, i]
            vt_norm = np.sqrt(vt_x**2 + vt_y**2)
            vt_x = vt_x / vt_norm
            vt_y = vt_y / vt_norm
            alpha = np.sqrt(v[i]**2 - orn_norm**2)
            f[i,0] = alpha * vt_x + orn_x
            f[i,1] = alpha * vt_y + orn_y

    return f