'''
    useful python packages
'''

# external python imports
import numpy as np
from scipy.fft import fft2, ifft2, fftfreq

# interal optical python module imports
from optical import medium
from optical.utils import wave_number
from optical.beam.propagation.boundary import absorbing as ABC
from optical.beam.propagation.boundary import transparent as TBC
from optical.beam.propagation.boundary import perfect as PML

'''
    spectral methods for estimate light beam propagation
'''

def split_step(
    field: np.ndarray,
    wave_length: float,
    medium: medium,
    region: tuple[np.ndarray, np.ndarray],
    zf: float,
    zi: float = 0.,
    iterations: int = 1,
    boundary: None | ABC = None
) -> np.ndarray:
    '''
        optical.beam.propagation.split_step
            estimate the propagated field along z axis through a medium
            using split-step beam propagation method.
    '''
    # compute transverse plane
    X, Y = region;                                              # cartesian coordinates
    Nx, Ny = field.shape;
    dx, dy = X[0, 1] - X[0,0], Y[1, 0] - Y[0, 0];
    # compute transverse Fourier plane
    k_x, k_y = 2. * np.pi * fftfreq(Nx, dx), 2. * np.pi * fftfreq(Ny, dy);
    k_x, k_y = np.meshgrid(k_x, k_y);
    # compute free-space propagation effects
    dz = (zf - zi) / float(iterations);
    Im_dz = 1.j * dz;
    _2k = 2. * wave_number(wave_length, n0 = medium.n0);
    exp_jH0dz = np.exp(Im_dz * (k_x**2. + k_y**2.) / _2k);
    # obtain boundary conditions
    S0 = boundary(X, Y) if isinstance(boundary, ABC) else np.zeros_like(field);
    # estimate propagation effects on light beam
    z = zi;
    for _ in range(iterations):
        # free space propagation
        field = ifft2(exp_jH0dz * fft2(field));
        # inhomogeneous and non-linear effects on propagation
        field *= np.exp(
            -Im_dz * (
                S0 + medium.non_linearity(field) + medium.inhomogeneity(X,Y,z)
            )
        );
        z += dz;
    return field;

def trotter_suzuki(
    field: np.ndarray,
    wave_length: float,
    medium: medium,
    region: tuple[np.ndarray, np.ndarray],
    zf: float,
    zi: float = 0.,
    iterations: int = 1,
    boundary: None | ABC = None
) -> np.ndarray:
    '''
        optical.beam.propagation.trotter_suzuki
            estimate the propagated field along z axis through a medium
            using trotter-suzuki method.
    '''
    X, Y = region;                                              # cartesian coordinates
    Nx, Ny = field.shape;
    dx, dy = X[0, 1] - X[0,0], Y[1, 0] - Y[0, 0];
    # compute transverse Fourier plane
    k_x, k_y = 2. * np.pi * fftfreq(Nx, dx), 2. * np.pi * fftfreq(Ny, dy);
    k_x, k_y = np.meshgrid(k_x, k_y);
    # compute free-space propagation effects
    dz = (zf - zi) / float(iterations);
    half_dz = dz;
    Im_dz = 1.j * half_dz / 2.;
    _2k = 2. * wave_number(wave_length, n0 = medium.n0);
    exp_jH0quarterdz = np.exp(Im_dz * (k_x**2. + k_y**2.) / _2k);
    exp_jH0halfdz = np.exp((Im_dz / 2.) * (k_x**2. + k_y**2.) / _2k);
    # obtain boundary conditions
    S0 = boundary(X, Y) if isinstance(boundary, ABC) else np.zeros_like(field);
    # estimate propagation effects on light beam
    z = zi;
    for _ in range(iterations):
        # free space propagation
        field = ifft2(exp_jH0halfdz * fft2(field));
        # inhomogeneous and non-linear effects on propagation
        field *= np.exp(
            -Im_dz * (
                S0 + medium.non_linearity(field) + medium.inhomogeneity(X,Y,z)
            )
        );
        z += half_dz;
        # free space propagation
        field = ifft2(exp_jH0quarterdz * fft2(field));
        # inhomogeneous and non-linear effects on propagation
        field *= np.exp(
            -Im_dz * (
                S0 + medium.non_linearity(field) + medium.inhomogeneity(X,Y,z)
            )
        );
        z += half_dz;
        # free space propagation
        field = ifft2(exp_jH0halfdz * fft2(field));
    return field;

'''
    finite diferences methods for estimate light beam propagation
'''

def crank_nicolson(
    field: np.ndarray,
    wave_length: float,
    medium: medium,
    region: tuple[np.ndarray, np.ndarray],
    zf: float,
    zi: float = 0.,
    iterations: int = 1,
    boundary: None | ABC | TBC | PML = None
) -> np.ndarray:
    '''
        optical.beam.propagation.crank_nicolson
            estimate the propagated field along z axis through a medium
            using crank-nicolson finite diferences method.
    '''

'''
    finite elements methods for estimate light beam propagation
'''
def finite_elements(
    field: np.ndarray,
    wave_length: float,
    medium: medium,
    region: tuple[np.ndarray, np.ndarray],
    zf: float,
    zi: float = 0.,
    iterations: int = 1,
    boundary: None | ABC | TBC | PML = None
) -> np.ndarray:
    '''
        optical.beam.propagation.finite_elements
            estimate the propagated field along z axis through a medium
            using finite elements method.
    '''