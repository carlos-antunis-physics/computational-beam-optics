'''
    importing of useful python packages
'''

import numpy as np

'''
    boundary conditions of optical beam propagation
'''

class absorbing:
    '''
        class optical.Propagation.boundary_condition.absorbing
            initializes an absorbing layer to act as boundary condition on
            propagation methods.
    '''
    widths: tuple[float, float];                # widths of absorbing layer
    __absorbance: float;                        # aborbance of the layer
    def __init__(self, widths: tuple[float, float], absorbance: float) -> None:
        # evaluate absorbing layer parameters
        self.widths = widths;                   # absorbing layer widths
        self.__absorbance = -1.j * absorbance;  # absorbance of layer
    def __call__(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        # evaluate parameters of computational window
        Lx, Ly = X[0,-1] - X[0,0], Y[-1,0] - Y[0,0];
        Lx, Ly = Lx / 2., Ly / 2.;
        wx, wy = self.widths;
        # construct absorbing gradative change in the refractive index
        abc = lambda u, l, w: np.where(u > l - w, ((u - (l - w)) / w) ** 2., 0.);
        return self.__absorbance * (
            abc(+X, Lx, wx) + abc(-X, Lx, wx) +
            abc(+Y, Ly, wy) + abc(-Y, Ly, wy)
        );
        
class transparent:
    '''
        class optical.Propagation.boundary_condition.transparent
            initializes a transparent boundary to act as boundary condition
            on propagation methods.
    '''
class perfectly_matched_layer:
    '''
        class optical.Propagation.boundary_condition.perfectly_matched_layer
            initializes a perfectly matched layer to act as boundary condi
            tion on propagation methods.
    '''