"""
Exemplify model errors, data errors, discretization errors,
and rounding errors in an exponential decay model.
"""
import matplotlib.pyplot as plt
import numpy as np
import sympy as sym

def model(t, I, a):
    return I*np.exp(-a*t)

def derive_true_solution():
    u = sym.symbols('u', cls=sym.Function)  # function u(t)
    t, a, p, I = sym.symbols('t a p I', real=True)

    def ode(u, t, a, p):
        """Define ODE: u' = (a + p*t)*u. Return residual."""
        return sym.diff(u, t) + (a + p*t)*u

    eq = ode(u(t), t, a, p)
    s = sym.dsolve(eq)
    # s is sym.Eq object u(t) == expression, we want u = expression,
    # so grab the right-hand side of the equality (Eq obj.)
    u = s.rhs
    print(u)
    # u contains C1, replace it with a symbol we can fit to
    # the initial condition
    C1 = sym.symbols('C1', real=True)
    u = u.subs('C1', C1)
    print(u)
    # Initial condition equation
    eq = u.subs(t, 0) - I
    s = sym.solve(eq, C1)  # solve eq wrt C1
    print(s)
    # s is a list s[0] = ...
    # Replace C1 in u by the solution
    u = u.subs(C1, s[0])
    print('u:', u)
    print(sym.latex(u))  # latex formula for reports

    # Consistency check: u must fulfill ODE and initial condition
    print('ODE is fulfilled:', sym.simplify(ode(u, t, a, p)))
    print('u(0)-I:', sym.simplify(u.subs(t, 0) - I))

    # Convert u expression to Python numerical function
    # (modules='numpy' allows numpy arrays as arguments,
    # we want this for t)
    u_func = sym.lambdify([t, I, a, p], u, modules='numpy')
    return u_func

true_model = derive_true_solution()

def model_errors():
    p_values = [0.01, 0.1, 1]
    a = 1
    I = 1
    t = np.linspace(0, 4, 101)
    legends = []
    # Work with figure(1) for the discrepancy and figure(2+i)
    # for plotting the model and the true model for p value no i
    for i, p in enumerate(p_values):
        u = model(t, I, a)
        u_true = true_model(t, I, a, p)
        discrepancy = u_true - u
        plt.figure(1)
        plt.plot(t, discrepancy)
        plt.figure(2+i)
        plt.plot(t, u, 'r-', t, u_true, 'b--')
        plt.legends.append('p=%g' % p)
    plt.figure(1)
    plt.legend(legends, loc='lower right')
    plt.savefig('tmp1.png'); plt.savefig('tmp1.pdf')
    for i, p in enumerate(p_values):
        plt.figure(2+i)
        plt.legend(['model', 'true model'])
        plt.title('p=%g' % p)
        plt.savefig('tmp%d.png' % (2+i)); plt.savefig('tmp%d.pdf' % (2+i))

def data_errors():
    N = 10000
    # Draw random numbers for I and a
    I_values = np.random.normal(1, 0.2, N)
    a_values = np.random.uniform(0.5, 1.5, N)
    # Compute corresponding u values for some t values
    t = [0, 1, 3]
    u_values = {}  # samples for various t values
    u_mean = {}
    u_std = {}
    for t_ in t:
        # Compute u samples corresponding to I and a samples
        u_values[t_] = [model(t_, I, a)
                        for I, a in zip(I_values, a_values)]
        u_mean[t_] = np.mean(u_values[t_])
        u_std[t_] = np.std(u_values[t_])

        plt.figure()
        dummy1, bins, dummy2 = np.hist(
            u_values[t_], bins=30, range=(0, I_values.max()),
            normed=True, facecolor='green')
        #plot(bins)
        plt.title('t=%g' % t_)
        plt.savefig('tmp_%g.png' % t_); plt.savefig('tmp_%g.pdf' % t_)
    # Table of mean and standard deviation values
    print('time   mean   st.dev.')
    for t_ in t:
        print('%3g    %.2f    %.3f' % (t_, u_mean[t_], u_std[t_]))

def solver(I, a, T, dt, theta):
    """Solve u'=-a*u, u(0)=I, for t in (0,T] with steps of dt."""
    dt = float(dt)            # avoid integer division
    Nt = int(round(T/dt))     # no of time intervals
    T = Nt*dt                 # adjust T to fit time step dt
    u = np.zeros(Nt+1)           # array of u[n] values
    t = np.linspace(0, T, Nt+1)  # time mesh
    u[0] = I                  # assign initial condition
    for n in range(0, Nt):    # n=0,1,...,Nt-1
        u[n+1] = (1 - (1-theta)*a*dt)/(1 + theta*dt*a)*u[n]
    return u, t

def discretization_errors():
    I = 1
    a = 1
    T = 4
    t = np.linspace(0, T, 101)
    schemes = {'FE': 0, 'BE': 1, 'CN': 0.5}  # theta to scheme name
    dt_values = [0.8, 0.4, 0.1, 0.01]
    for dt in dt_values:
        plt.figure()
        legends = []
        for scheme in schemes:
            theta = schemes[scheme]
            u, t = solver(I, a, T, dt, theta)
            u_e = model(t, I, a)
            error = u_e - u
            print('%s: dt=%.2f, %d steps, max error: %.2E' % \
                  (scheme, dt, len(u)-1, np.abs(error).max()))
            # Plot log(error), but exclude error[0] since it is 0
            plt.plot(t[1:], np.log(np.abs(error[1:])))
            legends.append(scheme)
        plt.xlabel('t');  plt.ylabel('log(abs(numerical error))')
        plt.legend(legends, loc='upper right')
        plt.title(r'$\Delta t=%g$' % dt)
        plt.savefig('tmp_dt%g.png' % dt); plt.savefig('tmp_dt%g.pdf' % dt)

def solver_decimal(I, a, T, dt, theta):
    """Solve u'=-a*u, u(0)=I, for t in (0,T] with steps of dt."""
    from decimal import Decimal as D
    dt = D(dt)
    a = D(a)
    theta = D(theta)
    Nt = int(round(D(T)/dt))
    T = Nt*dt
    u = np.zeros(Nt+1, dtype=object)  # array of Decimal objects
    t = np.linspace(0, float(T), Nt+1)
    u[0] = D(I)               # assign initial condition
    for n in range(0, Nt):    # n=0,1,...,Nt-1
        u[n+1] = (1 - (1-theta)*a*dt)/(1 + theta*dt*a)*u[n]
    return u, t

def rounding_errors(I=1, a=1, T=4, dt=0.1):
    import decimal
    digits_values = [4, 16, 64, 128]
    # "Exact" arithmetics is taken as 1000 decimals here
    decimal.getcontext().prec = 1000
    u_e, t = solver_decimal(I=I, a=a, T=T, dt=dt, theta=0.5)
    for digits in digits_values:
        decimal.getcontext().prec = digits  # set no of digits
        u, t = solver_decimal(I=I, a=a, T=T, dt=dt, theta=0.5)
        error = u_e - u
        error = np.array(error[1:], dtype=float)
        print('%d digits, %d steps, max abs(error): %.2E' % \
              (digits, len(u)-1, np.abs(error).max()))

if __name__ == '__main__':
    #model_errors()
    #data_errors()
    discretization_errors()
    #rounding_errors()
    #rounding_errors(dt=0.001)
    #rounding_errors(I=1000, a=100, T=0.04, dt=0.001)
    #rounding_errors(I=1000, a=100, T=0.04, dt=0.000001)
    plt.show()
