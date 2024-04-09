import numpy as np
from scipy.fftpack import fft2,ifft2,fftshift

def splitstepPropagate(
    U: np.ndarray | dict[any, np.ndarray],
    wavelength: float | list[float],
    region: tuple[np.ndarray, np.ndarray],
    Z: np.ndarray,
    medium: dict
) -> np.ndarray | dict[any, np.ndarray]:
    # compute general parameters of propagation method
    k0 = 2.0 * np.pi / np.asarray(wavelength);
    k = k0 * medium['base refractive index'];

    # compute direct space parameters of propagation procedure
    X, Y = region;
    Nx, Ny = len(X[0,:]), len(Y[:,0]);
    Lx, Ly = X[0,-1] - X[0,0], Y[-1,0] - Y[0,0];
    dx, dy, dz = Lx / (Nx - 1), Ly / (Ny - 1), Z[1] - Z[0];

    # compute k space parameters of propagation procedure
    kx, ky = (
        [np.pi / (Nx * dx) * (2.0 * (m - 0.5) - (Nx - 1)) for m in range(Nx)],
        [np.pi / (Ny * dy) * (2.0 * (m - 0.5) - (Ny - 1)) for m in range(Ny)]
    )
    Kx, Ky = np.meshgrid(kx,ky);

    # compute parameters of propagation procedure
    idz = 1.0j * dz;
    H = fftshift(                   # free space transfer function
        np.exp(-idz / (2.0 * k) * (Kx ** 2 + Ky ** 2))
    );

    U[:, -1:0] = 0.0;
    U[-1:0, :] = 0.0;

    # estimate optical field propagation
    for z in Z:
        # evaluate free space propagation effects in k-space
        U = ifft2(H * fft2(U));
        # evaluate nonlinear propagation effects in direct space
        U = np.exp(-idz * medium['non-linearity'](U)) * U;
    return U;
