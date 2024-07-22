'''
    useful packages importing
'''

import numpy as np
from scipy.fft import fft2, ifft2, fftshift

from .. import Medium as medium

'''
    boundary conditions
'''

# absorbing boundary condition
class absorbing:
    '''
        class optical.beam.propagation.absorbing
            creates an absorbing layer as boundary condition to propagation methods.
    '''
    __wx: float;                                # x axis width of absorbing layer
    __wy: float;                                # y axis width of absorbing layer
    __absorbance: float;                        # absorbance of absorbing layer
    def __init__(
        self,
        widths: tuple[float, float],            # x and y axis widths of boundary absorbing layer
        absorbance: float                       # absorbance of absorbing layer
    ) -> None:
        # evaluate absorbing layer parameters
        self.__wx, self.__wy = widths;          # absorbing layer widths
        self.__absorbance = 1.j * absorbance;   # absorbance of layer
    def __call__(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        '''
            evaluate the absorbing layer effects on computational window.
        '''
        # evaluate parameters of computational window
        Lx, Ly = X[0,-1] - X[0,0], Y[-1,0] - Y[0,0];
        Lx, Ly = Lx / 2., Ly / 2.;
        # construct absorbing layer
        ABS = lambda u, L, w: np.where(u > L - w, ((u - (L - w)) / w) ** 2., 0.);
        return -self.__absorbance * (
            ABS(+X,Lx,self.__wx) +
            ABS(-X,Lx,self.__wx) +
            ABS(+Y,Ly,self.__wy) +
            ABS(-Y,Ly,self.__wy)
        );

'''
    computational estimation of light beam propagation
'''

# split-step propagation method
def split_step_propagate(
    U: np.ndarray,
    region: tuple[np.ndarray, np.ndarray],
    z: np.ndarray,
    medium: medium,
    wave_length: float,
    boundary_condition: absorbing | None = None
) -> np.ndarray:
    # evaluate coordinates on computational window
    X, Y = region;                              # x, y meshgrid of coordinates on region
    Nx, Ny = U.shape;
    Lx, Ly = X[0,-1] - X[0,0], Y[-1,0] - Y[0,0];
    dz = z[1] - z[0];
    Im_dz = 1.j * dz;
    # evaluate Fourier plane coordinates
    kx = np.linspace(-np.floor(Nx / 2), +np.floor(Nx / 2), Nx) * np.pi / Lx;
    ky = np.linspace(-np.floor(Ny / 2), +np.floor(Ny / 2), Ny) * np.pi / Ly;
    Kx, Ky = np.meshgrid(kx, ky);
    # compute general parameters of propagation
    k0 = 2. * np.pi / wave_length;
    Im_dz_by_2k = Im_dz / (2. * k0 * medium.n0);
    H = fftshift(np.exp(Im_dz_by_2k * (Kx ** 2. + Ky ** 2.)));
    if boundary_condition == None:
        for _z in z:
            # evaluate free space propagation effects in Fourier plane
            U = ifft2(H * fft2(U));
            # evaluate inhomogeneity and non-linear effects in transverse plane
            S = medium.apply_nonlinearity(U) + medium.apply_refractive_index(X,Y,_z);
            U = np.exp(-Im_dz * S) * U;
    elif isinstance(boundary_condition, absorbing):
        S0 = boundary_condition(X, Y);          # insert absorbing layer effects
        for _z in z:
            # evaluate free space propagation effects in Fourier plane
            U = ifft2(H * fft2(U));
            # evaluate inhomogeneity and non-linear effects in transverse plane
            S = S0 + medium.apply_nonlinearity(U) + medium.apply_refractive_index(X,Y,_z);
            U = np.exp(-Im_dz * S) * U;
    return U;