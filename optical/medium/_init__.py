import numpy as np

def create(
    n0: float,
    nonLinearity: np.ufunc = lambda U: 0.0
) -> dict:
    '''
    ## (function) `optical.medium.create`:
        creates an optical medium.

    ### syntax:
            `medium = optical.medium.create(n0)`
        
    ### parameters:
            `n0`: `float`
                base refractive index of medium.
            `nonLinearity`: `np.ufunc`
                term in helmholtz equation which represents non-linear behavior.
    '''
    return {
        'base refractive index': n0,
        'non-linearity': nonLinearity
    };
