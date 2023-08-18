import numpy as np


def differentiate(u, dt):
    N = len(u)
    du = np.zeros(N)
    du[1:-1] = (u[2:]-u[:-2])/(2*dt)
    du[0] = (u[1]-u[0])/dt
    du[-1] = (u[-1]-u[-2])/dt
    return du

def test_differentiate():
    t = np.linspace(0, 1, 10)
    u = t**2
    due = 2*t
    du = differentiate(u, t[1]-t[0])
    assert np.linalg.norm((du-due)[1:-1], np.inf) < 1e-12

if __name__ == '__main__':
    test_differentiate()


