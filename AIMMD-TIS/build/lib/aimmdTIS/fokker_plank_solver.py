import numpy as np 
from tqdm import tqdm
""" 
Fokker Plank Solver for the committor in 2D potentials. 
Provided by Roberto Covino and Gianmarco Lazzeri from Frankfurt Institute of Advanced Studies.
Using Fokker Plank solver method from: https://doi.org/10.1063/1.5118362
"""

# def interpolate(x, y, P, X, Y):
#     """
#     Interpolate P values on a X, Y grid.
#     """
#     k = ((y - Y[0, 0]) / (Y[1, 1] - Y[0, 0]))
#     h = ((x - X[0, 0]) / (X[1, 1] - X[0, 0]))


#     # # Ensure k and h are within the bounds of the grid
#     # if k < 0 or k >= len(Y) - 1 or h < 0 or h >= len(X) - 1:
#     #     return None  # Return None or handle out-of-bounds case as desired
    
#     k = min(max(0, k), len(Y) - 1)
#     h = min(max(0, h), len(X) - 1)

#     k1, k2 = int(k), int(k) + 1
#     a2 = k - int(k)
#     a1 = 1 - a2
#     h1, h2 = int(h), int(h) + 1
#     b2 = h - int(h)
#     b1 = 1 - b2
#     return (
#                    P[k1, h1] * a1 * b1 +
#                    P[k1, h2] * a1 * b2 +
#                    P[k2, h1] * a2 * b1 +
#                    P[k2, h2] * a2 * b2
#            ) / (a1 * b1 + a1 * b2 + a2 * b1 + a2 * b2)

def interpolate(x, y, P, X, Y):
    """
    Perform bilinear interpolation to interpolate P values on an X, Y grid.

    Parameters:
    x (float): X-coordinate where interpolation is performed.
    y (float): Y-coordinate where interpolation is performed.
    P (ndarray): Grid of values to be interpolated.
    X (ndarray): Grid representing X-coordinates.
    Y (ndarray): Grid representing Y-coordinates.

    Returns:
    float: Interpolated value.
    """
    x0, x1 = X[0, 0], X[-1, -1]
    y0, y1 = Y[0, 0], Y[-1, -1]

    # Check if the point lies within the bounds of the grid
    if x < x0 or x > x1 or y < y0 or y > y1:
        raise ValueError("Point lies outside the grid bounds.")

    # Find indices for interpolation
    i, j = np.searchsorted(X[0], x), np.searchsorted(Y[:, 0], y)

    if i == 0:
        i = 1
    elif i == len(X[0]):
        i = len(X[0]) - 1

    if j == 0:
        j = 1
    elif j == len(Y):
        j = len(Y) - 1

    # Bilinear interpolation
    x_ratio = (x - X[0, i - 1]) / (X[0, i] - X[0, i - 1])
    y_ratio = (y - Y[j - 1, 0]) / (Y[j, 0] - Y[j - 1, 0])

    Q11, Q21 = P[j - 1, i - 1], P[j, i - 1]
    Q12, Q22 = P[j - 1, i], P[j, i]

    interpolated_value = (Q11 * (1 - x_ratio) * (1 - y_ratio) +
                          Q21 * x_ratio * (1 - y_ratio) +
                          Q12 * (1 - x_ratio) * y_ratio +
                          Q22 * x_ratio * y_ratio)

    return interpolated_value

def solve_committor_by_relaxation(
        X, Y, Fx, Fy, A, B, P0, progress=[5, 4, 2, 1]):
    """
    Compute committor in 2D with relaxation method (Brownian dynamics).

    Parameters
    ----------
    X: x-coordinates on a 2D grid
    Y: y-coordinates on a 2D grid
    Fx: force's x component on a 2D grid
    Fy: force's y component on a 2D grid
    A: points in A on a 2D grid
    B: points in B on a 2D grid
    P0: initial guess for committor on a 2D grid
    progress: iteratively increase the resolution based on the vector's
values

    Returns
    -------
    P0: committor estimate
    """
    for split in tqdm(progress, position=0):
        X1 = X[::split, ::split]
        Y1 = Y[::split, ::split]
        P1 = P0[::split, ::split]
        Fx1 = Fx[::split, ::split]
        Fy1 = Fy[::split, ::split]
        A1 = A[::split, ::split]
        B1 = B[::split, ::split]
        dFx = np.diff(X1, axis=1)[1:-1, :-1] * Fx1[1:-1, 1:-1]
        dFy = np.diff(Y1, axis=0)[:-1, 1:-1] * Fy1[1:-1, 1:-1]
        dFx[dFx > +1.] = +1.
        dFx[dFx < -1.] = -1.
        dFy[dFy > +1.] = +1.
        dFy[dFy < -1.] = -1.
        r = np.max(np.abs(
            (((P1[2:, 1:-1] + P1[:-2, 1:-1] - 2 * P1[1:-1, 1:-1]) +
              (P1[1:-1, 2:] + P1[1:-1, :-2] - 2 * P1[1:-1, 1:-1])) +
             (dFx * (P1[2:, 1:-1] - P1[:-2, 1:-1]) +
              dFy * (P1[1:-1, 2:] - P1[1:-1, :-2])) / 2)
        ))
        while True:
            r1 = 0 + r
            for i in range(100):
                P1[1:-1, 1:-1] = (2 * (P1[2:, 1:-1] + P1[:-2, 1:-1] +
                                       P1[1:-1, 2:] + P1[1:-1, :-2]) +
                               (dFy * (P1[2:, 1:-1] - P1[:-2, 1:-1]) +
                                dFx * (P1[1:-1, 2:] - P1[1:-1, :-2]))) / 8
                P1[:, 0] = P1[:, 1]
                P1[:, -1] = P1[:, -2]
                P1[0, :] = P1[1, :]
                P1[-1, :] = P1[-2, :]
                P1[P1 < 0] = 0
                P1[P1 > 1] = 1
                P1[A1] = 0
                P1[B1] = 1
            r = np.max(np.abs(((
                 (P1[2:, 1:-1] + P1[:-2, 1:-1] - 2 * P1[1:-1, 1:-1]) +
                 (P1[1:-1, 2:] + P1[1:-1, :-2] - 2 * P1[1:-1, 1:-1])) +
                 (dFx * (P1[2:, 1:-1] - P1[:-2, 1:-1]) +
                  dFy * (P1[1:-1, 2:] - P1[1:-1, :-2])) / 2)))
            if np.abs(r - r1) < 1e-16:
                break
        P0 = np.array([interpolate(x, y, P1, X1, Y1)
                       for x, y in zip(X.ravel(),
                        Y.ravel())]).reshape(X.shape)
    return P0
