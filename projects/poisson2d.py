import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import scipy.sparse as sparse
from sympy.utilities.lambdify import implemented_function
from poisson import Poisson

x, y = sp.symbols('x,y')

class Poisson2D:
    r"""
    Solve Poisson's equation in 2D::

       \nabla^2 u(x, y) = f(x, y), x, y in [0, Lx] x [0, Ly]

    with homogeneous Dirichlet boundary conditions.
    """

    def __init__(self, Lx, Ly, Nx, Ny, ue):
        self.px = Poisson(Lx, Nx) # we can reuse some of the code from the 1D case with self.px.* or self.py.* 
        self.py = Poisson(Ly, Ny)
        self.create_mesh()
        self.ue = ue

    def create_mesh(self):
        """Return a 2D Cartesian mesh
        """
        #self.py.d
        #self.Lx,self.Ly = self.px.L, self.py.L
        #self.Nx,self.Ny = self.px.N, self.py.N
        #self.dx,self.dy = self.Lx / self.Nx, self.Ly / self.Ny 
        #x, y = np.linspace(0,self.Lx,self.Nx+1), np.linspace(0,self.Ly,self.Ny+1)
        #x = np.linspace(0,self.px.L,self.px.N + 1)
        #y = np.linspace(0,self.py.L,self.py.N + 1)
        xi = self.px.create_mesh(self.px.N)
        yj = self.py.create_mesh(self.py.N)
        self.xij,self.yij = np.meshgrid(xi,yj,indexing='ij',sparse=True)
        return self.xij, self.yij #np.meshgrid(xi,yj,indexing='ij',sparse=True)

    def laplace(self):
        """
        Return a vectorized Laplace operator. Uses an implementation of the D2-matrix which is scaled with
        the increment dx or dy
        """
        D2x = self.px.D2() #(1./self.px.dx**2) * self.px.D2()#self.px.N)
        D2y = self.py.D2() #(1./self.py.dx**2) * self.py.D2()#self.py.N)
        Ix, Iy = sparse.eye(self.px.N+1), sparse.eye(self.py.N+1)
        
        return (sparse.kron(D2x, Iy) + sparse.kron(Ix, D2y)).tolil()
        #return (sparse.kron(D2x, sparse.eye(self.py.N+1)) + sparse.kron(sparse.eye(self.px.N+1), D2y)).tolil()

    def assemble(self, f=None):
        """
        Return assemble coefficient matrix A and right hand side vector b
        """
        D = self.laplace()
    
        # Boundary matrix and boundary indices
        #B = np.ones((self.px.N+1,self.py.N+1),dtype=bool)
        #B[1:-1,1:-1] = 0
        #bnds = np.where(B.ravel() == 1)[0]
        bnds = self.get_boundary_idx()
        for i in bnds:
            D[i] = 0
            D[i,i] = 1
        D = D.tocsr()

        b = np.zeros((self.px.N+1, self.py.N+1))
        b[:,:] = self.meshfunc(f) #sp.lambdify((x,y),f)(self.xij,self.yij)
        uij = self.meshfunc(self.ue) #sp.lambdify((x,y),self.ue)(self.xij,self.yij)
        b.ravel()[bnds] = uij.ravel()[bnds]

        return D, b
    
    def get_boundary_idx(self):
        B = np.ones((self.px.N+1,self.py.N+1),dtype=bool)
        B[1:-1,1:-1] = 0

        return np.where(B.ravel() == 1)[0]

    def meshfunc(self,u):

        return sp.lambdify((x,y),u)(self.xij,self.yij)

    def l2_error(self, u):
        """Return l2-error

        Parameters
        ----------
        u : array
            The numerical solution (mesh function)
        ue : Sympy function
            The analytical solution
        """
        uij = self.meshfunc(self.ue) #sp.lambdify((x,y),self.ue)(self.xij,self.yij)
        return np.sqrt(self.px.dx*self.py.dx * np.sum((u - uij)**2))

    def __call__(self):#, f=implemented_function('f', lambda x, y: 2)(x, y)):
        """Solve Poisson's equation with a given righ hand side function

        Parameters
        ----------
        N : int
            The number of uniform intervals
        f : Sympy function
            The right hand side function f(x, y)

        Returns
        -------
        The solution as a Numpy array

        """
        #self.xij,self.yij = self.create_mesh()
        A, b = self.assemble(f=sp.diff(self.ue, x, 2) + sp.diff(self.ue,y,2))
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
    Lx, Ly = 1., 1.
    Nx, Ny = 100, 100
    ue = x**3 + y**2 #sp.exp(4*sp.cos(x) + sp.sin(y))
    bc = (ue.subs(x,0),ue.subs(x,Lx),ue.subs(y,0),ue.subs(y,Ly))
    print(ue)
    d2_ue = ue.diff(x, 2) + ue.diff(y,2)
    sol = Poisson2D(Lx=Lx,Ly=Ly,Nx=Nx,Ny=Ny,ue=ue)
    u = sol()#f=d2_ue)
    print('L2-error: %g' %(sol.l2_error(u)))
