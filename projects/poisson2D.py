import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import scipy.sparse as sparse
from sympy.utilities.lambdify import implemented_function
from poisson import Poisson

x, y = sp.symbols('x,y')

# Below we create a solver that reuses some of the implementation from 
# the 1D solver in poisson.py. 

class Poisson2D:
    r"""Solve Poisson's equation in 2D::

        \nabla^2 u(x, y) = f(x, y), x, y in [0, Lx] x [0, Ly]

    with homogeneous Dirichlet boundary conditions.
    """

    def __init__(self, Lx, Ly, Nx, Ny, ue):
        # Note that the best solution for a solver may not be to have Nx and 
        # Ny in the __init__ function, because it should be possible to modify the
        # number of points without creating a new class.
        self.px = Poisson(Lx, Nx) # we can reuse some of the code from the 1D case
        self.py = Poisson(Ly, Ny)
        self.create_mesh()
        self.ue = ue

    def create_mesh(self):
        """Return a 2D Cartesian mesh
        """
        xi = self.px.create_mesh(self.px.N)
        yj = self.py.create_mesh(self.py.N)
        self.xij, self.yij = np.meshgrid(xi, yj, indexing='ij', sparse=True)
        return self.xij, self.yij

    def laplace(self):
        """Return a vectorized Laplace operator"""
        D2x = self.px.D2()
        D2y = self.py.D2()
        Ix = sparse.eye(self.px.N+1)
        Iy = sparse.eye(self.py.N+1)
        return (sparse.kron(D2x, Iy) + sparse.kron(Ix, D2y)).tolil()

    def assemble(self, f=None):
        """Return assembled coefficient matrix A and right hand side vector b"""
        A = self.laplace()
        bnds = self.get_boundary_indices()
        for i in bnds:
            A[i] = 0
            A[i, i] = 1
        A = A.tocsr()
        b = np.zeros((self.px.N+1, self.py.N+1))
        b[:, :] = self.meshfunction(f)
        # Set boundary conditions
        uij = self.meshfunction(self.ue)
        b.ravel()[bnds] = uij.ravel()[bnds]
        return A, b

    def meshfunction(self, u):
        """Return Sympy function as mesh function

        Parameters
        ----------
        u : Sympy function

        Returns
        -------
        array - The input function as a mesh function
        """
        return sp.lambdify((x, y), u)(self.xij, self.yij)

    def get_boundary_indices(self):
        """Return indices of vectorized matrix that belongs to the boundary"""
        B = np.ones((self.px.N+1, self.py.N+1), dtype=bool)
        B[1:-1, 1:-1] = 0
        return np.where(B.ravel() == 1)[0]

    def l2_error(self, u):
        """Return l2-error

        Parameters
        ----------
        u : array
            The numerical solution (mesh function)
        """
        return np.sqrt(self.px.dx*self.py.dx*np.sum((u - self.meshfunction(self.ue))**2))

    def __call__(self):
        """Solve Poisson's equation with a given manufactured solution

        Returns
        -------
        The solution as a Numpy array

        """
        A, b = self.assemble(f=sp.diff(self.ue, x, 2)+sp.diff(self.ue, y, 2))
        return sparse.linalg.spsolve(A, b.ravel()).reshape((self.px.N+1, self.py.N+1))

def convergence_rates(ue, m=6):
    E = []
    h = []
    N0 = 8
    for m in range(m):
        sol = Poisson2D(1, 1, N0, N0, ue)
        u = sol()
        E.append(sol.l2_error(u))
        h.append(sol.px.dx)
        N0 *= 2
    r = [np.log(E[i-1]/E[i])/np.log(h[i-1]/h[i]) for i in range(1, m+1, 1)]
    return r, np.array(E), np.array(h)

def test_poisson2d():
    r, E, h = convergence_rates(sp.exp(sp.cos(4*sp.pi*x)*sp.sin(2*sp.pi*y)))
    assert abs(r[-1]-2) < 1e-2

if __name__ == '__main__':
    test_poisson2d()