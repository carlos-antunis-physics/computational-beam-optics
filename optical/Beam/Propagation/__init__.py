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
    computational estimators of light beam propagation
'''

def split_step(
    field: np.ndarray,
    region: tuple[np.ndarray, np.ndarray],
    medium: medium,
    wave_length: float,
    zf: float,
    z0: float = 0.,
    iterations: int = 1,
    boundary: None | ABC = None
) -> np.ndarray:
    '''
        optical.Propagation.split_step
            estimate the field propagated along z axis through a medium
            using split-step beam propagation.
    '''
    # evaluate transverse plane parameters
    X, Y = region;                              # transverse plane cartesian meshgrid
    Nx, Ny = field.shape;
    Lx, Ly = X[0,-1] - X[0,0], Y[-1,0] - Y[0,0];
    # evaluate Fourier plane parameters
    _r = {'x': 1 - (Nx % 2), 'y': 1 - (Ny % 2)};
    nu_x = np.linspace(-np.floor(Nx/2), +np.floor((Nx - _r['x'])/2), Nx) / Lx;
    nu_y = np.linspace(-np.floor(Ny/2), +np.floor((Ny - _r['y'])/2), Ny) / Ly;
    nu_x, nu_y = np.meshgrid(nu_x, nu_y);
    # evaluate propagation parameters
    dz = (zf - z0) / float(iterations);
    Im_dz = 1.j * dz;
    Imdz_pi_lambda = Im_dz * np.pi * (wave_length / medium.n0);
    FS_prop = fftshift(np.exp(Imdz_pi_lambda * (nu_x ** 2. + nu_y ** 2.)));
    # apply specified boundary conditions
    S0 = boundary(X,Y) if isinstance(boundary, ABC) else np.zeros_like(field);
    # estimate propagation effects on light beam transverse profile
    z = z0;
    for iteration in range(iterations):
        z += iteration * dz;
        # evaluate free space propagation effects in Fourier plane
        field = ifft2(FS_prop * fft2(field));
        # evluate inhomogeneity and non-linear effects in transverse plane
        field *= np.exp(-Im_dz * (S0 + medium.non_linearity(field) + medium(X,Y,z)));
        # ensure zero at boundaries
        field[-1:0,:] = 0.; field[:,-1:0] = 0.;
    return field;

def trotter_suzuki(
    field: np.ndarray,
    region: tuple[np.ndarray, np.ndarray],
    medium: medium,
    wave_length: float,
    zf: float,
    z0: float = 0.,
    iterations: int = 1,
    boundary: None | ABC = None
) -> np.ndarray:
    '''
        optical.Propagation.trotter_suzuki
            estimate the field propagated along z axis through a medium
            using trotter-suzuki aproximation of beam propagation.
    '''
    # evaluate transverse plane parameters
    X, Y = region;                              # transverse plane cartesian meshgrid
    Nx, Ny = field.shape;
    Lx, Ly = X[0,-1] - X[0,0], Y[-1,0] - Y[0,0];
    # evaluate Fourier plane parameters
    _r = {'x': 1 - (Nx % 2), 'y': 1 - (Ny % 2)};
    nu_x = np.linspace(-np.floor(Nx/2), +np.floor((Nx - _r['x'])/2), Nx) / Lx;
    nu_y = np.linspace(-np.floor(Ny/2), +np.floor((Ny - _r['y'])/2), Ny) / Ly;
    nu_x, nu_y = np.meshgrid(nu_x, nu_y);
    # evaluate propagation parameters
    dz = (zf - z0) / float(iterations);
    Im_dz = 1.j * dz; half_dz = dz / 2.;
    Imhalf_dz = 1.j * half_dz;
    Imhalfdz_pi_lambda = Imhalf_dz * np.pi * (wave_length / medium.n0);
    FS_prop = fftshift(np.exp(Imhalfdz_pi_lambda * (nu_x ** 2. + nu_y ** 2.)));
    # apply specified boundary conditions
    S0 = boundary(X,Y) if isinstance(boundary, ABC) else np.zeros_like(field);
    # estimate propagation effects on light beam transverse profile
    z = z0;
    for iteration in range(iterations):
        # evluate inhomogeneity and non-linear effects in transverse plane
        z += iteration * half_dz;
        field *= np.exp(-Im_dz * (S0 + medium.non_linearity(field) + medium(X,Y,z)));
        # evaluate free space propagation effects in Fourier plane
        field = ifft2(FS_prop * fft2(field));
        # evluate inhomogeneity and non-linear effects in transverse plane
        z += iteration * half_dz;
        field *= np.exp(-Im_dz * (S0 + medium.non_linearity(field) + medium(X,Y,z)));
        # ensure zero at boundaries
        field[-1:0,:] = 0.; field[:,-1:0] = 0.;
    return field;

def crank_nicolson(
    field: np.ndarray,
    region: tuple[np.ndarray, np.ndarray],
    medium: medium,
    wave_length: float,
    zf: float,
    z0: float = 0.,
    iterations: int = 1,
    boundary: None | ABC | TBC | PML = None
) -> np.ndarray:
    '''
        optical.Propagation.crank_nicolson
            estimate the field propagated along z axis through a medium
            using crank-nicolson finite diferences scheme.
    '''
    raise NotImplementedError(
        "This numerical method hasn't been implemented yet."
    );
