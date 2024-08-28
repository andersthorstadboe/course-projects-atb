import numpy as np

def mesh_function(f, t):

    fun = np.zeros_like(t)
    for i in range(len(t)):
        fun[i] = f(t[i])

    return fun

def mesh_function1(f, t):

    fun = np.zeros_like(t)
    for i, ti in enumerate(t):
        fun[i] = f(ti)

    return fun

def func(t):
    if t >= 0 and t <= 3:
        return np.exp(-t)
    elif t > 3 and t <= 4:
        return np.exp(-3*t)
    else :
        raise RuntimeError(f"Wrong input t = {t}") 

def test_mesh_function():
    t = np.array([1, 2, 3, 4])
    f = np.array([np.exp(-1), np.exp(-2), np.exp(-3), np.exp(-12)])
    fun = mesh_function(func, t)
    assert np.allclose(fun, f)
