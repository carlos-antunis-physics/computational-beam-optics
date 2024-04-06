import numpy as np

def create(
    n0: float
) -> dict:
    '''
    ## (function) `optical.medium.create`:
        creates an optical medium.

    ### syntax:
            `medium = optical.medium.create(n0)`
        
    ### parameters:
            `refractiveIndex`: `float`
                base refractive index of medium.
    '''
    return {
        'base refractive index': n0
    };