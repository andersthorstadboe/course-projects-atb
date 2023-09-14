import numpy as np
import sympy as sp
from scipy import sparse
import matplotlib.pyplot as plt

x, t, c, L = sp.symbols('x,t,c,L')

class Wave1D:
    """Class for solving the wave equation

    Parameters
    ----------
    N : int
        Number of uniform spatial intervals
    L0 : number
        The extent of the domain, which is [0, L]
    c0 : number, optional
        The wavespeed
    cfl : number, optional
        CFL number
    u0 : Sympy function of x, t, c and L
        Used for specifying initial condition
    """
    def __init__(self, N, L0=1, c0=1, cfl=1, u0=sp.exp(-200*(x-L/2+c*t)**2)):
        self.N = N
        self.L = L0
        self.c = c0
        self.cfl = cfl
        self.x = np.linspace(0, L0, N+1)
        self.dx = L0 / N
        self.u0 = u0
        self.unp1 = np.zeros(N+1)
        self.un = np.zeros(N+1)
        self.unm1 = np.zeros(N+1)

    def D2(self, bc):
        """Return second order differentiation matrix

        Paramters
        ---------
        bc : int
            Boundary condition

        Note
        ----
        The returned matrix is not divided by dx**2
        """
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (self.N+1, self.N+1), 'lil')
        if bc == 1: # Neumann condition is baked into stencil
            raise NotImplementedError

        elif bc == 3: # periodic (Note u[0] = u[-1])
            raise NotImplementedError

        return D

    def apply_bcs(self, bc, u=None):
        """Apply boundary conditions to solution vector

        Parameters
        ----------
        bc : int
            Boundary condition in space
            - 0 Dirichlet
            - 1 Neumann
            - 2 Open boundary
            - 3 periodic
        u : array, optional
            The solution array to fix at boundaries
            If not provided, use self.unp1

        """
        u = u if u is not None else self.unp1
        if bc == 0: # Dirichlet condition
            u[0] = 0
            u[-1] = 0

        elif bc == 1: # Neumann condition
            pass

        elif bc == 2: # Open boundary
            raise NotImplementedError

        elif bc == 3:
            raise NotImplementedError

        else:
            raise RuntimeError(f"Wrong bc = {bc}")

    @property
    def dt(self):
        return self.cfl*self.dx/self.c

    def __call__(self, Nt, cfl=None, bc=0, ic=0, save_step=100):
        """Solve wave equation

        Parameters
        ----------
        Nt : int
            Number of time steps
        cfl : number
            CFL number
        bc : int, optional
            Boundary condition in space
            - 0 Dirichlet
            - 1 Neumann
            - 2 Open boundary
            - 3 periodic
        ic : int, optional
            Initial conditions
            - 0 Specify un = u(x, t=0) and unm1 = u(x, t=-dt)
            - 1 Specify un = u(x, t=0) and u_t(x, t=0) = 0
        save_step : int, optional
            Save solution every save_step time step

        Returns
        -------
        Dictionary with key, values as timestep, array of solution
        The number of items in the dictionary is Nt/save_step, and
        each value is an array of length N+1

        """
        D = self.D2(bc)
        self.cfl = C = self.cfl if cfl is None else cfl
        dt = self.dt
        u0 = sp.lambdify(x, self.u0.subs({L: self.L, c: self.c, t: 0}))

        # First step. Set un and unm1
        self.unm1[:] = u0(self.x) # unm1 = u(x, 0)
        plotdata = {0: self.unm1.copy()}
        if ic == 0: # use sympy function for un = u(x, dt)
            u0 = sp.lambdify(x, self.u0.subs({L: self.L, c: self.c, t: dt}))
            self.un[:] = u0(self.x)

        else: # use u_t = 0 for un = u(x, dt)
            self.un[:] = self.unm1 + 0.5*C**2* (D @ self.unm1)
            self.apply_bcs(bc, self.un)
        if save_step == 1:
            plotdata[1] = self.un.copy()

        for n in range(2, Nt+1):
            self.unp1[:] = 2*self.un - self.unm1 + C**2 * (D @ self.un)
            self.apply_bcs(bc)
            self.unm1[:] = self.un
            self.un[:] = self.unp1
            if n % save_step == 0: # save every save_step timestep
                plotdata[n] = self.unp1.copy()

        return plotdata

    def plot_with_offset(self, data):
        Nd = len(data)
        v = np.array(list(data.values()))
        t = np.array(list(data.keys()))
        dt = t[1]-t[0]
        v0 = abs(v).max()
        fig = plt.figure(facecolor='k')
        ax = fig.add_subplot(111, facecolor='k')
        for i, u in data.items():
            ax.plot(self.x, u+i*v0/dt, 'w', lw=2, zorder=i)
            ax.fill_between(self.x, u+i*v0/dt, i*v0/dt, facecolor='k', lw=0, zorder=i-1)
        plt.show()

    def animation(self, data):
        from matplotlib import animation
        fig, ax = plt.subplots()
        v = np.array(list(data.values()))
        t = np.array(list(data.keys()))
        save_step = t[1]-t[0]
        line, = ax.plot(self.x, data[0])
        ax.set_ylim(v.min(), v.max())
        def update(frame):
            line.set_ydata(data[frame*save_step])
            return (line,)
        ani = animation.FuncAnimation(fig=fig, func=update, frames=len(data), blit=True)
        ani.save('wavemovie.apng', writer='pillow', fps=5) # This animated png opens in a browser
        ani.to_jshtml()
        plt.show()

def test_pulse_bcs():
    sol = Wave1D(100, cfl=1, L0=2, c0=1)
    data = sol(100, bc=0, ic=0, save_step=100)
    assert np.linalg.norm(data[0]+data[100]) < 1e-12
    data = sol(100, bc=0, ic=1, save_step=100)
    assert np.linalg.norm(data[0]+data[100]) < 1e-12
    data = sol(100, bc=1, ic=0, save_step=100)
    assert np.linalg.norm(data[0]-data[100]) < 1e-12
    data = sol(100, bc=1, ic=1, save_step=100)
    assert np.linalg.norm(data[0]-data[100]) < 1e-12
    data = sol(100, bc=2, ic=0, save_step=100)
    assert np.linalg.norm(data[100]) < 1e-12
    data = sol(100, bc=2, ic=1, save_step=100)
    assert np.linalg.norm(data[100]) < 1e-12
    data = sol(100, bc=3, ic=0, save_step=100)
    assert np.linalg.norm(data[0]-data[100]) < 1e-12
    data = sol(100, bc=3, ic=1, save_step=100)
    assert np.linalg.norm(data[0]-data[100]) < 1e-12

if __name__ == '__main__':
    #sol = Wave1D(100, cfl=1, L0=2, c0=1)
    #data = sol(100, bc=3, save_step=1, ic=1)
    #sol.animation(data)
    test_pulse_bcs()
    #data = sol(200, bc=2, ic=0, save_step=100)

