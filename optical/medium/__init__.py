import numpy as np
from dataclasses import dataclass
from .waveguides import waveguide

@dataclass
class __medium__:
    n0: float;
    nonlinearity: np.ufunc;
    waveguides: list[waveguide];

def create(
    n0: float | np.ufunc,
    nonlinearity: np.ufunc = lambda U: 0.0,
    waveguides: list[waveguide] = []
) -> __medium__:
    '''
    ## `optical.medium.create`:
        creates an optical medium with `n0` as refractive index.
    
    ### syntax:
        `medium = optical.medium.create(n0)`
    
    ### parameters:
        `n0`: `float`
            base refractive index of the medium.
        [optional] `nonlinearity`: `numpy.ufunc`
            nonlinear behavior of the medium.
        [optional] `waveguides`: `list[waveguides.waveguide]`
            waveguide written on the medium
    '''
    return __medium__(
        n0 = n0,
        nonlinearity = nonlinearity,
        waveguides = waveguides
    );