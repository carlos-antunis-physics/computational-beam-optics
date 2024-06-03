import numpy as np
from scipy.fft import fft2, ifft2, fftshift
from .. import *

def split_step(
    U: np.ndarray,
    wavelength: np.float128,
    region: tuple[np.ndarray, np.ndarray],
    z: np.ndarray,
    medium: medium
) -> np.ndarray:
    '''
        ## `optical.beam.propagation.split_step`
            solve nonlinear inhomogeneous paraxial wave equation in (2+1)
            dimensions using BPM split step spectral method.
    
        ### syntax
            `optical.beam.propagation.split_step(U,lambda,(X,Y),z,medium)`
    '''
    # compute coordinates
    X, Y = region;                  # x, y axis meshgrids
    Nx, Ny = X.shape;
    Lx, Ly = X[0,-1] - X[0,0], Y[-1,0] - Y[0,0];
    dx, dy, dz = Lx / (Nx - 1), Ly / (Ny - 1), z[1] - z[0];
    idz = 1.0j * dz;
    _idz = -idz;

    # compute k-coordinates
    kx = [np.pi / (Nx * dx) * float(2 * (m - 0.5) - (Nx - 1)) for m in range(Nx)];
    ky = [np.pi / (Ny * dy) * float(2 * (m - 0.5) - (Ny - 1)) for m in range(Ny)];
    Kx, Ky = np.meshgrid(kx, ky);

    # compute general parameters of propagation
    k0 = 2.0 * np.pi / wavelength;
    _idz_by_2k = _idz / (2.0 * k0 * medium.refIndex);
    H = fftshift(np.exp(_idz_by_2k * (Kx ** 2. + Ky ** 2.)));

    # estimate propagation
    for Z in z:
        # evaluate free space propagation effects in Fourier plane
        U = ifft2(H * fft2(U));
        # evaluate nonlinear and inhomogeneity effects in direct space
        S = medium.apply_nonlinearity(U);
        for waveguide in medium.waveguides:
            S += waveguide.apply_refractive_index(X,Y,Z);
        U = np.exp(_idz * S) * U;
    return U;