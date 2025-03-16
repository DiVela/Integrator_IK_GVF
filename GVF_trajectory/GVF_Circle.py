import numpy as np

class gvf_circumference():
    def __init__(self, center, rad):
        self.x0 = center[0]
        self.y0 = center[0]
        self.radius = rad
    
    def gen_circumference_points(self, pts):
        alpha = np.linspace(0, 2*np.pi, pts)
        x = self.radius * np.cos(alpha) + self.x0
        y = self.radius * np.sin(alpha) + self.y0
        return np.array([x, y])
    
    def phi(self, p):
        phi=(p[0] - self.x0)**2 + (p[1] - self.y0)**2 - self.radius**2
        return phi
    
    def grad_phi(self, p):
        grad = np.zeros((2,1))
        grad[0] = 2 * (p[0] - self.x0)
        grad[1] = 2 * (p[1] - self.y0)
        return grad

    def hess_phi(self):
        H = np.zeros((2,2))
        H[0,0] = 2 
        H[0,1] = 0
        H[1,0] = 0
        H[1,1] = 2
        return H
