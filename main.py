import numpy as np

def least_squares_approximation(x, f):
    """
    Compute the presolved linear least squares approximation of the function f(x).

    Args:
        x: A numpy array of x values
        f: A numpy array of f(X) values

    Returns:
        A numpy array of the coefficients of the least squares approximation
    """

    k = len(x)

    #Compute the s_x_i and s_x_2_i values
    s_x_i = np.sum(x)
    s_x_2_i = np.sum(x**2)
    sfi = np.sum(f)
    s_x_i_f_i = np.sum(x*f)

    #Compute c1
    c1 = (s_x_2_i*sfi - s_x_i*s_x_i_f_i)/(k*s_x_2_i - s_x_i**2)

    #c0 computation
    c0 = ((s_x_2_i*sfi)-(s_x_i*s_x_i_f_i))/(k*s_x_2_i - s_x_i**2)

    return np.array([c0, c1])

    