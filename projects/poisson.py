import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import scipy.sparse as sparse
from sympy.utilities.lambdify import implemented_function

x = sp.Symbol('x')

class Poisson:
    """Solve Poisson's equation in 1D::

        u''(x) = f(x), x in [0, L], u(0) = a, u(L) = b

    where a and b are numbers.

    """
    def __init__(self, L=1, N=None):
        self.L = L
        self.N = N
        self.ue = None
        if isinstance(N, int):
            self.create_mesh(N)

    def D2(self):
        """Return second order differentiation matrix
        """
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (self.N+1, self.N+1), 'lil')
        D[0, :4] = 2, -5, 4, -1
        D[-1, -4:] = -1, 4, -5, 2
        D /= self.dx**2
        return D

    def assemble(self, bc=(0, 0), f=None):
        """Assemble coefficient matrix and right hand side vector

        Parameters
        ----------
        bc : 2-tuple of numbers
            The boundary conditions at x=0 and x=L
        f : Sympy Function
            The right hand side as a Sympy function

        Returns
        -------
        A : scipy sparse matrix
            Coefficient matrix
        b : 1D array
            Right hand side vector
        """
        D = self.D2()
        D[0, :4] = 1, 0, 0, 0
        D[-1, -4:] = 0, 0, 0, 1
        b = np.zeros(self.N+1)
        b[1:-1] = sp.lambdify(x, f)(self.x[1:-1])
        b[0] = bc[0]
        b[-1] = bc[1]
        return D.tocsr(), b

    def create_mesh(self, N):
        """Uniform discretization of the line [0, L]

        Parameters
        ----------
        N : int
            The number of uniform intervals

        Returns
        -------
        x : array
            The mesh
        """
        self.N = N
        self.dx = self.L / N
        self.x = np.linspace(0, self.L, self.N+1)
        return self.x

    def __call__(self, N, bc=(0, 0), f=implemented_function('f', lambda x: 2)(x)):
        """Solve Poisson's equation

        Parameters
        ----------
        N : int
            The number of uniform intervals
        bc : 2-tuple of numbers
            The boundary conditions at x=0 and x=L
        f : Sympy Function
            The right hand side as a Sympy function

        Returns
        -------
        The solution as a Numpy array

        """
        self.create_mesh(N)
        A, b = self.assemble(bc=bc, f=f)
        return sparse.linalg.spsolve(A, b)

    def l2_error(self, u, ue):
        """Return l2-error

        Parameters
        ----------
        u : array_like
            Numerical solution
        ue : Sympy function
            The analytical solution
        Returns
        -------
        The l2-error as a number

        """
        uj = sp.lambdify(x, ue)(self.x)
        return np.sqrt(self.dx*np.sum((uj-u)**2))

def test_poisson(L):
    sol = Poisson(L=2,N=100)

    N = 100
    ue = x**2; tol = 1e-12
    bc=(ue.subs(x,0),ue.subs(x,L))
    data = sol(100,bc=(ue.subs(x,0),ue.subs(x,L)),f=sp.diff(ue, x, 2))
    print('\nManufactured solution: ', ue)
    print(f'Boundary conditions: u(0)={bc[0]:2.4f}, u(L)={bc[1]:2.2f}')
    print('Test with N = %g; tol = %g' %(N,tol))
    print('Test L2-error: %g\n' %(sol.l2_error(data,ue)))
    assert sol.l2_error(data,ue) < tol

    sol = Poisson(L=2,N=100)
    N = 30000
    ue = sp.exp(4*sp.cos(x)); tol = 1e-5
    bc=(ue.subs(x,0),ue.subs(x,L))
    data = sol(N,bc=(ue.subs(x,0),ue.subs(x,L)),f=sp.diff(ue, x, 2))
    print('Manufactured solution: ', ue)
    print(f'Boundary conditions: u(0)={bc[0]:2.4f}, u(L)={bc[1]:2.2f}')
    print('Test with N = %g; tol = %g' %(N,tol))
    print('Test L2-error: %g\n' %(sol.l2_error(data,ue)))
    assert sol.l2_error(data,ue) < tol

    sol = Poisson(L=2,N=100)
    N = 10000
    ue = sp.exp(x**2); tol = 1e-5
    bc=(ue.subs(x,0),ue.subs(x,L))
    data = sol(N,bc=(ue.subs(x,0),ue.subs(x,L)),f=sp.diff(ue, x, 2))
    print('Manufactured solution: ', ue)
    print(f'Boundary conditions: u(0)={bc[0]:2.4f}, u(L)={bc[1]:2.2f}')
    print('Test with N = %g; tol = %g' %(N,tol))
    print('Test L2-error: %g\n' %(sol.l2_error(data,ue)))
    assert sol.l2_error(data,ue) < tol

if __name__ == '__main__':
    L = 2
    sol = Poisson(L=L)
    ue = 0*x#sp.exp(4*sp.cos(x))
    #ue = x**2
    bc = (ue.subs(x, 0), ue.subs(x, L))
    u = sol(100, bc=bc, f=sp.diff(ue, x, 2))
    print('Manufactured solution: ', ue)
    print(f'Boundary conditions: u(0)={bc[0]:2.4f}, u(L)={bc[1]:2.2f}')
    print(f'Discretization: N = {sol.N}')
    print(f'L2-error: {sol.l2_error(u, ue)}')
    test_poisson(L=L)
