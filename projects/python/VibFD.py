"""
In this module we study the vibration equation

    u'' + w^2 u = f, t in [0, T]

where w is a constant and f(t) is a source term assumed to be 0.
We use various boundary conditions.

"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse

class VibSolver:
    """
    Solve vibration equation::

        u'' + w**2 u = f,

    """
    def __init__(self, Nt, T, w=0.35, I=1):
        """
        Parameters
        ----------
        Nt : int
            Number of time steps
        T : float
            End time
        I, w : float, optional
            Model parameters
        """
        self.I = I
        self.w = w
        self.T = T
        self.set_mesh(Nt)

    def set_mesh(self, Nt):
        """Create mesh of chose size

        Parameters
        ----------
        Nt : int
            Number of time steps
        """
        self.Nt = Nt
        self.dt = self.T/Nt
        self.t = np.linspace(0, self.T, Nt+1)

    def u_exact(self):
        """Exact solution of the vibration equation

        Returns
        -------
        ue : array_like
            The solution at times n*dt
        """
        return self.I*np.cos(self.w*self.t)

    def l2_error(self):
        """Compute the l2 error norm of solver

        Returns
        -------
        float
            The l2 error norm
        """
        u = self()
        ue = self.u_exact()
        return np.sqrt(np.trapz((ue-u)**2, self.t))

    def convergence_rates(self, m=4, N0=32):
        """
        Compute convergence rate

        Parameters
        ----------
        m : int
            The number of mesh sizes used
        N0 : int
            Initial mesh size

        Returns
        -------
        r : array_like
            The m-1 computed orders
        E : array_like
            The m computed errors
        dt : array_like
            The m time step sizes
        """
        E = []
        dt = []
        self.set_mesh(N0) # Set initial size of mesh
        for m in range(m):
            self.set_mesh(2*self.Nt)
            E.append(self.l2_error())
            dt.append(self.dt)
        r = [np.log(E[i-1]/E[i])/np.log(dt[i-1]/dt[i]) for i in range(1, m+1, 1)]
        return r, np.array(E), np.array(dt)

    def test_order(self):
        r, E, dt = self.convergence_rates(4)
        assert np.allclose(np.array(r), self.order, atol=1e-2)

class VibHPL(VibSolver):
    """
    Second order accurate recursive solver

    Boundary conditions u(0)=I and u'(0)=0
    """
    order = 2

    def __call__(self):
        u = np.zeros(self.Nt+1)
        u[0] = self.I
        u[1] = u[0] - 0.5*self.dt**2*self.w**2*u[0]
        for n in range(1, self.Nt):
            u[n+1] = 2*u[n] - u[n-1] - self.dt**2*self.w**2*u[n]
        return u

class VibFD2(VibSolver):
    """
    Second order accurate solver using boundary conditions::

        u(0)=I and u(T)=I

    The boundary conditions require that T = n*pi/w, where n is an even integer.
    """
    order = 2

    def __init__(self, Nt, T=2*np.pi, w=0.35, I=1):
        VibSolver.__init__(self, Nt, T, w, I)
        T = T * w / np.pi
        assert T.is_integer() and T % 2 == 0

    def __call__(self):
        D2 = sparse.diags([np.ones(self.Nt), np.full(self.Nt+1, -2), np.ones(self.Nt)],
                          np.array([-1, 0, 1]), (self.Nt+1, self.Nt+1), 'lil')
        D2 *= (1/self.dt**2)
        A = D2 + self.w**2*sparse.eye(self.Nt+1)
        A[0, :4] = 1, 0, 0, 0
        A[-1, -4:] = 0, 0, 0, 1
        b = np.zeros(self.Nt+1)
        b[0] = self.I
        b[-1] = self.I
        u = sparse.linalg.spsolve(A, b)
        return u

class VibFD4(VibFD2):
    """
    Fourth order accurate solver using boundary conditions::

        u(0)=I and u(T)=I

    The boundary conditions require that T = n*pi/w, where n is an even integer.
    """
    order = 4

    def __call__(self):
        D2 = sparse.diags([-np.ones(self.Nt-1), np.full(self.Nt, 16), np.full(self.Nt+1, -30), np.full(self.Nt, 16), -np.ones(self.Nt-1)],
                          np.array([-2, -1, 0, 1, 2]), (self.Nt+1, self.Nt+1), 'lil')
        #D2[1, :5] = np.array([11, -20, 6, 4, -1])
        #D2[-2, -5:] = np.array([11, -20, 6, 4, -1])[::-1]
        #D2[1, :7] = np.array([-13, 93, -285, 470, -255, -147, 137]) / 15
        #D2[-2, -7:] = np.array([-13, 93, -285, 470, -255, -147, 137])[::-1] / 15
        #D2[1, :6] = np.array([10, -15, -4, 14, -6, 1])
        #D2[-2, -6:] = np.array([10, -15, -4, 14, -6, 1])[::-1]
        D2[1, :7] = np.array([0, 45, -154, 214, -156, 61, -10]) # findiff
        D2[-2, -7:] = np.array([0, 45, -154, 214, -156, 61, -10])[::-1]
        D2 *= (1/(12*self.dt**2))
        A = D2 + self.w**2*sparse.eye(self.Nt+1)
        A[0, :4] = 1, 0, 0, 0
        A[-1, -4:] = 0, 0, 0, 1
        b = np.zeros(self.Nt+1)
        b[0] = self.I
        b[-1] = self.I
        u = sparse.linalg.spsolve(A, b)
        #print(np.linalg.cond(A.toarray()))
        return u

def test_order():
    w = 0.35
    VibHPL(32, 2*np.pi/w, w).test_order()
    VibFD2(32, 2*np.pi/w, w).test_order()
    VibFD4(32, 2*np.pi/w, w).test_order()

if __name__ == '__main__':
    test_order()
