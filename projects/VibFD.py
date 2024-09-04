"""
In this module we study the vibration equation

    u'' + w^2 u = f, t in [0, T]

where w is a constant and f(t) is a source term assumed to be 0.
We use various boundary conditions.

"""
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy import sparse

t = sp.Symbol('t')

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

    def ue(self):
        """Return exact solution as sympy function
        """
        return self.I*sp.cos(self.w*t)

    def u_exact(self):
        """Exact solution of the vibration equation

        Returns
        -------
        ue : array_like
            The solution at times n*dt
        """
        return sp.lambdify(t, self.ue())(self.t)

    def l2_error(self):
        """Compute the l2 error norm of solver

        Returns
        -------
        float
            The l2 error norm
        """
        u = self()
        ue = self.u_exact()
        return np.sqrt(self.dt*np.sum((ue-u)**2))

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
            self.set_mesh(self.Nt+10)
            E.append(self.l2_error())
            dt.append(self.dt)
        r = [np.log(E[i-1]/E[i])/np.log(dt[i-1]/dt[i]) for i in range(1, m+1, 1)]
        return r, np.array(E), np.array(dt)

    def test_order(self, m=5, N0=100, tol=0.1):
        r, E, dt = self.convergence_rates(m, N0)
        assert np.allclose(np.array(r), self.order, atol=tol)

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

    def __init__(self, Nt, T, w=0.35, I=1):
        VibSolver.__init__(self, Nt, T, w, I)
        T = T * w / np.pi
        assert T.is_integer() and T % 2 == 0

    def assemble_matrix(self):
        g = self.w**2 * self.dt**2
        D2 = sparse.diags([1, -2, 1], np.array([-1, 0, 1]),(self.Nt+1, self.Nt+1), format='lil')
        Id = sparse.eye(self.Nt+1,format='lil')
        A  = (D2 + g*Id).tolil()
        b  = np.zeros(self.Nt+1)
        return A, b

    def __call__(self):
        
        u = np.zeros(self.Nt+1)
        A, b = self.assemble_matrix()
        # Boundary values
        A[0,:3] = 1, 0, 0
        A[-1,-3:] = 0, 0, 1
        b[0], b[-1] = self.I, self.I # Boundary condition for solution vector

        u = sparse.linalg.spsolve(A.tocsr(),b)

        return u

class VibFD3(VibFD2):
    """
    Second order accurate solver using mixed Dirichlet and Neumann boundary
    conditions::

        u(0)=I and u'(T)=0

    The boundary conditions require that T = n*pi/w, where n is an even integer.
    """
    order = 2

    def __init__(self, Nt, T, w=0.35, I=1):
        VibSolver.__init__(self, Nt, T, w, I)
        T = T * w / np.pi
        assert T.is_integer() and T % 2 == 0

    def __call__(self):
        u  = np.zeros(self.Nt+1)
        A, b = self.assemble_matrix()
        A[0,:3] = 1, 0, 0
        A[-1,-3:] = 1/(2*self.dt), -4/((2*self.dt)), 3/(2*self.dt)

        b  = np.zeros(self.Nt+1)
        b[0] = self.I    # Boundary condition for solution vector
        u = sparse.linalg.spsolve(A.tocsr(),b)
        
        return u

class VibFD4(VibFD2):
    """
    Fourth order accurate solver using boundary conditions::

        u(0)=I and u(T)=I

    The boundary conditions require that T = n*pi/w, where n is an even integer.
    """
    order = 4

    def assemble_matrix4(self):
        g = self.w**2 * 12*self.dt**2
        D2 = sparse.diags([-1, 16, -30, 16, -1], np.array([-2, -1, 0, 1, 2]),(self.Nt+1, self.Nt+1), format='lil')
        D2[1,:6] = np.array([10, -15, -4, 14, -6, 1])
        #D2[-2,:6] = np.array([10, -15, -4, 14, -6, 1])[::-1]
        D2[-2,-6:] = 1, -6, 14, -4, -15, 10
        Id = sparse.eye(self.Nt+1,format='lil')
        A  = (D2 + g*Id).tolil()
        b  = np.zeros(self.Nt+1)

        return A, b 

    def __call__(self):
        u = np.zeros(self.Nt+1)
        A, b = self.assemble_matrix4()
        # Boundary values
        A[0,:6] = 1, 0, 0, 0, 0, 0
        A[-1,:6] = 0, 0, 0, 0, 0, 1
        b[0], b[-1] = self.I, self.I

        u = sparse.linalg.spsolve(A.tocsr(),b) 
        return u
    
class VibMMS(VibFD2):
    """ Second order accurate solver using
        Method of Manufactured solutions (MMS),
        solving 
        u'' + w^2u = f
        with 
        f_1 = t^4
        f_2 = exp(sin(t))  
    """
    order = 2

    def __init__(self, Nt, T, w=0.35, I=1):
        VibSolver.__init__(self, Nt, T, w, I)
        #T = T * w / np.pi
        #assert T.is_integer() and T % 2 == 0

    def u1(self):
        #return t**2
        return sp.exp(sp.sin(t))
        
    def assemble_matrixMMS(self):
        g = self.w**2 * self.dt**2
        D2 = sparse.diags([1, -2, 1], np.array([-1, 0, 1]),(self.Nt+1, self.Nt+1), format='lil')
        Id = sparse.eye(self.Nt+1,format='lil')
        A  = (D2 + g*Id).tolil()

        ue1 = self.u1()
        f = ue1.diff(t,2) + self.w**2 * ue1
        b = self.dt**2 * (sp.lambdify(t,f)(self.t))

        return A, b
    
    def __call__(self):
        u = np.zeros(self.Nt+1)
        A, b = self.assemble_matrixMMS()

        # Boundary conditions
        A[0,:3]   = np.array([1, 0, 0])
        A[-1,-3:] = np.array([0, 0, 1])
        b[0]  = self.u1().subs(t, 0)
        b[-1] = self.u1().subs(t,self.T)

        u = sparse.linalg.spsolve(A.tocsr(),b)
        return u

def test_order():
    w = 0.35; T = 2.0
    VibHPL(8, 2*np.pi/w, w).test_order()
    VibFD2(8, 2*np.pi/w, w).test_order()
    VibFD3(8, 2*np.pi/w, w).test_order()
    VibFD4(8, 2*np.pi/w, w).test_order(N0=20, tol=0.515e1)
    VibMMS(8, T,         w).test_order(tol=0.25e1)

if __name__ == '__main__':
    test_order()
