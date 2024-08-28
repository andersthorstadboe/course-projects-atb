import numpy as np


def theta_solver(I, a, T, dt, theta):
    """Solving u' = -a*u, u(0) = I, for t in (0,T] with steps of dt"""
    dt = float(dt)
    Nt = int(round(T/dt))
    T  = Nt*dt
    u  = np.zeros(Nt+1)
    t  = np.linspace(0, T, Nt+1)

    u[0]  = I                                       # Initial condition
    u[1:] = ((1 - (1-theta)*a*dt)/(1 + theta*dt*a))
    u[:]  = np.cumprod(u)

    return u, t

def u_exact(t, I, a):
    return I*np.exp(-a*t)

def amp_u(theta, adt):
    return (1 - (1-theta)*adt)/(1+ theta*adt)

def amp_exact(adt):
    return np.exp(-adt)

import matplotlib.pyplot as plt

def compare_num_exact(theta, I, a, T, dt):

    t_e = np.linspace(0,T,1001)  # Mesh for exact solution
    u_e = u_exact(t_e,I,a)       # Exact solution

    fig,ax = plt.subplots(1,1)
    fig.suptitle(r'Exp. growth, different $\theta$, dt = %g' %(dt))

    for th in theta:
        u, t = theta_solver(I=I, a=a, T=T, dt=dt, theta=th)

        ax.plot(t,u,'--o',label=[r'$\theta$ = %g' %(th)])
        
    
    ax.plot(t_e,u_e, 'k--',label=['exact'])
    ax.legend()
    ax.set_xlabel('t'); ax.set_ylabel('u')
    ax.grid()

    #plt.show()

def amp_compare(theta,adt):
    
    A_exact = amp_exact(adt)

    fig,ax = plt.subplots(1,1)
    fig.suptitle('Amp.factor')

    for th in theta:
        A_num   = amp_u(theta=th,adt=adt)

        ax.plot(adt,A_num, '--o',label=[r'$\theta$ = %g' %(th)])

    ax.plot(adt,A_exact,'k--o',label='exact')
    ax.legend()    
    ax.set_xlabel('t'); ax.set_ylabel('u')
    ax.grid()

    
if __name__ == '__main__':
    compare_num_exact(theta=[0.0, 0.5, 1.0], I=0.1, a=-1, T=4.0, dt=0.1)
    amp_compare(theta=[0.0, 0.5, 1.0], adt=np.linspace(0,3,10))

    plt.show()
    