'''
    importing of useful python packages
'''

import numpy as np

'''
    importing of fortran linear algebra utils
'''

# import methods from linear_algebra.f95
from optical.Utils import linear_algebra

'''
    python interface for linear_algebra.f95 routines
'''
# thomas algorithm for solving linear system of equations
thomas = lambda ld, d, ud, b: linear_algebra.linear_algebra.thomas(
    n = np.array(d).shape[0],
    lower_diagonal = ld, diagonal = d, upper_diagonal = ud,
    b = b
);