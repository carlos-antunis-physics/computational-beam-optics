'''
    importing of useful python packages
'''

import numpy as np
from scipy.fft import fft2, ifft2, fftshift

'''
    importing of optical module utils
'''

# import medium object class
from optical import medium
# import wave number evaluator
from optical import wave_number
# import thomas algorithm for solving linear systems
from optical.Utils import thomas
# import boundary conditions for optical propagation methods
from .boundary import absorbing as ABC
from .boundary import transparent as TBC
from .boundary import perfectly_matched_layer as PML

'''
    computational estimation of light beam propagation
'''

def split_step(
    field: np.ndarray,
    region: tuple[np.ndarray, np.ndarray],
    z: np.array,
    medium: medium,
    wave_length: float,
    boundary: ABC | None = None
) -> np.ndarray:
    '''
        optical.Propagation.split_step
            estimate the field propagated along z points on medium by
            split-step beam propagation method.
    '''
    # evaluate simulation parameters
    X, Y = region;                              # region meshgrid of cartesian coordinates
    Nx, Ny = field.shape;
    Lx, Ly = X[0,-1] - X[0,0], Y[-1,0] - Y[0,0];
    dx, dy, dz = X[0,1] - X[0,0], Y[1,0] - Y[0,0], z[-1] - z[0];
    Im_dz = 1.j * dz;
    # evaluate fourier plane parameters
    if Nx % 2 == 0:
        nu_x = np.linspace(-np.floor(Nx/2.), +np.floor((Nx - 1.)/2.), Nx) / Lx;
    else:
        nu_x = np.linspace(-np.floor(Nx/2.), +np.floor(Nx/2.), Nx) / Lx;
    if Ny % 2 == 0:
        nu_y = np.linspace(-np.floor(Ny/2.), +np.floor((Ny - 1.)/2.), Ny) / Ly;
    else:
        nu_y = np.linspace(-np.floor(Ny/2.), +np.floor(Ny/2.), Ny) / Ly;
    nu_x, nu_y = np.meshgrid(nu_x, nu_y);
    Im_pi_wvl = Im_dz * np.pi * (wave_length / medium.n0);
    free_space_prop = fftshift(np.exp(Im_pi_wvl * (nu_x ** 2. + nu_y ** 2.)));
    # evaluate propagation with refered boundary conditions
    if boundary == None:
        S = np.zeros_like(field); S0 = S;
        for _z in z:
            # evaluate free space propagation effects in Fourier plane
            field = ifft2(free_space_prop * fft2(field));
            # evaluate inhomogeneity and non-linear effects in transverse plane
            S = S0 + medium.non_linearity(field) + medium(X,Y,_z);
            field = np.exp(-Im_dz * S) * field;
            # ensure zero at boundaries
            field[-1:0,:] = 0.; field[:,-1:0] = 0.;
        return field;
    elif isinstance(boundary, ABC):
        S = np.zeros_like(field);
        S0 = boundary(X, Y);                    # insert absorbing layer effects
        for _z in z:
            # evaluate free space propagation effects in Fourier plane
            field = ifft2(free_space_prop * fft2(field));
            # evaluate inhomogeneity and non-linear effects in transverse plane
            S = S0 + medium.non_linearity(field) + medium(X,Y,_z);
            field = np.exp(-Im_dz * S) * field;
            # ensure zero at boundaries
            field[-1:0,:] = 0.; field[:,-1:0] = 0.;
        return field;

def crank_nicolson(
    field: np.ndarray,
    region: tuple[np.ndarray, np.ndarray],
    z: np.array,
    medium: medium,
    wave_length: float,
    boundary: ABC | TBC | PML | None = None
) -> np.ndarray:
    '''
        optical.Propagation.crank_nicolson
            estimate the field propagated along z points on medium by crank-nicolson
            scheme for finite differences on beam propagation method.
    '''
    # evaluate simulation parameters
    X, Y = region;                              # region meshgrid of cartesian coordinates
    dx, dy, dz = X[0,1] - X[0,0], Y[1,0] - Y[0,0], z[-1] - z[0];