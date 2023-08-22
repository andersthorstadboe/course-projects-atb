import numpy as np


def differentiate(u, dt):
    N = len(u)
    du = np.zeros(N)
    du[0] = (u[1]-u[0])/dt
    du[-1] = (u[-1]-u[-2])/dt
    for i in range(1, N-1):
        du[i] = (u[i+1] - u[i-1]) / (2*dt)
    return du

def differentiate_vector(u, dt):
    N = len(u)
    du = np.zeros(N)
    du[1:-1] = (u[2:]-u[:-2])/(2*dt)
    du[0] = (u[1]-u[0])/dt
    du[-1] = (u[-1]-u[-2])/dt
    return du

def test_differentiate():
    t = np.linspace(0, 1, 10)
    dt = t[1] - t[0]
    u = t**2
    du1 = differentiate(u, dt)
    du2 = differentiate_vector(u, dt)
    assert np.allclose(du1, du2)

if __name__ == '__main__':
    test_differentiate()
