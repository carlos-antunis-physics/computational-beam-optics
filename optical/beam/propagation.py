import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift
import sys
sys.path.append('..')
from medium import __medium__
from medium.waveguides import waveguide

def splitstep(
    U: np.ndarray,
    wavelength: float,
    region: tuple[np.ndarray, np.ndarray],
    Z: np.ndarray,
    medium: __medium__,
) -> np.ndarray:
    # compute direct space parameters of propagation
    X, Y = region;
    Nx, Ny = len(X[0,:]), len(Y[0])
    Lx, Ly = X[0, -1] - X[0, 0], Y[-1, 0] - Y[0, 0];
    dx, dy, dz = Lx / (Nx - 1), Ly / (Ny - 1), Z[1] - Z[0];
    idz = -1.0j * dz;

    # compute k-space parameters of propagation
    kx, ky = (
        [np.pi / (float(Nx - 1) * dx) * float(2 * m - (Nx - 1)) for m in range(Nx)],
        [np.pi / (float(Ny - 1) * dy) * float(2 * m - (Ny - 1)) for m in range(Ny)],
    );
    Kx, Ky = np.meshgrid(kx, ky);

    # compute general parameters of propagation
    k0 = 2.0 * np.pi / wavelength;
    idz_ov_2k = idz / (2.0 * k0 * medium.n0);
    H = fftshift(np.exp(-idz_ov_2k * (Kx ** 2 + Ky ** 2)));

    # estimate optical field propagation
    for z in Z:
        U[:, -1:0] = U[-1:0, :] = 0.0;
        # evaluate free space propagation efects in k-space
        U = ifft2(H * fft2(U));
        # evaluate nonlinear effects in direct space
        U = np.exp(idz * medium.nonlinearity(U)) * U;
        # evaluate waveguides effects in direct space
        Delta_n = np.zeros(X.shape)
        for WG in medium.waveguides:
            Delta_n += WG.apply_refractive_index((X,Y), z);
        U = np.exp(idz * Delta_n) * U;
    return U;
