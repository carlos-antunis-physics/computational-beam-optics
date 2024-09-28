'''
    useful python packages
'''

# external python imports
import numpy as np

'''
    boundary conditions for estimation of optical beam propagation
'''

class absorbing:
    '''
        class optical.beam.propagation.absorbing
    ''' 
    __widths: tuple[float, float];
    __absorbance: float;
    def __init__(self, widths: tuple[float, float], absorbance: float) -> None:
        self.__widths = widths;
        self.__absorbance = -1.j * absorbance;
    def __call__(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        Lx, Ly = X[0, -1] - X[0, 0], Y[-1, 0] - Y[0, 0];
        half_Lx, half_Ly = Lx / 2., Ly / 2.;
        wx, wy = self.__widths;
        grad = lambda u, l, w: np.where(u > l - w, ((u - (l - w)) / w) ** 2., 0.);
        return self.__absorbance * (
            grad(+X, half_Lx, wx) + grad(-X, half_Lx, wx) +
            grad(+Y, half_Ly, wy) + grad(-Y, half_Ly, wy)
        );

class transparent:
    '''
        class optical.beam.propagation.transparent
    '''

class perfect:
    '''
        class optical.beam.propagation.perfect
    '''