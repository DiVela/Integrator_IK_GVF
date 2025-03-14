import numpy as np

def control_law(states, N, B):

    #------------- Consenso ----------------
    f = np.zeros((N,2))
    L = B @ B.T
    theta = np.zeros((N,1))
    for i in range(N):
        theta[i] = np.atan2(states[i, 1], states[i, 0])

    u = - L @ theta

    



    print(u)
        

        
    return f