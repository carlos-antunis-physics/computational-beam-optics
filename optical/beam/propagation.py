import numpy as np
from scipy.fft import fft2, ifft2, fftshift
from .. import medium

'''
    absorbing boundaries construction.
'''

class absorbing_boundary:
    def __init__(
        self,
        lengths: tuple[np.float128, np.float128],
        widths: tuple[np.float128, np.float128],
        absorbance: np.float128,
        center: tuple[np.float128, np.float128] = (0.,0.)
    ) -> None:
        '''
        ## `optical.beam.propagation.absorbing_boudary`
            constructs a `widths` sized rectangular absorbing layer with informed
            `absorbance` for refered `lengths` of computational window, as the
            boundary condition to propagation methods.

        ### syntax
            `U = optical.beam.propagation.absorbing_boundary((Lx,Ly), (wx, wy), alpha)`
        #### optional parameters
            `center`: `tuple[numpy.float128, numpy.float128]
                cartesian coordinates of computational window center.
        '''
        # compute absorbing layer sizes
        Lx, Ly = lengths;
        self.__Lx, self.__Ly = Lx / 2., Ly / 2.;
        self.__w_x, self.__w_y = widths;
        self.__x0, self.__y0 = center;

        # construct a waveguide in boundaries with refered refractive index
        self.__absorbance = absorbance;
    def apply(
        self,
        X: np.ndarray,
        Y: np.ndarray
    ) -> np.ndarray:
        '''
        ## `[optical.beam.propagation.absorbing_boundary] AB.apply`
            obtain the absorbance of boundary in (`X`,`Y`).

        ### syntax
            `AB.apply(X,Y)`
        '''
        # inform absorbing refractive index shape
        __abc = lambda coordinate, L, width: np.where(
            # any coordinate in absorbing layer
            ((coordinate) > (L - width)),
            # refractive index with increasing absorbing refractive index
            ((coordinate - (L - width)) / width) ** 3.,
            0.                      # and zero elsewhere
        );

        # compute absorbance matrix
        return 1.0j * self.__absorbance * (
          __abc(+(X - self.__x0), self.__Lx, self.__w_x) +
          __abc(-(X - self.__x0), self.__Lx, self.__w_x) +
          __abc(+(Y - self.__y0), self.__Ly, self.__w_y) +
          __abc(-(Y - self.__y0), self.__Ly, self.__w_y)
        );

'''
    split-step propagation method.
'''

def split_step(
    U: np.ndarray,
    wavelength: np.float128,
    region: tuple[np.ndarray, np.ndarray],
    z: np.ndarray,
    medium: medium,
    boundary_condition: None | absorbing_boundary = None
) -> np.ndarray:
    '''
        ## `optical.beam.propagation.split_step`
            solve nonlinear inhomogeneous paraxial wave equation in (2+1)
            dimensions using BPM split step spectral method.

        ### syntax
            `optical.beam.propagation.split_step(U,lambda,(X,Y),z,medium)`
        #### optional parameters
            `boundary_condition`:
              `optical.beam.propagation.absorbing_boundary`
                absorbing boundary condition to use in propagation.
    '''
    # compute coordinates
    X, Y = region;                  # x, y axis meshgrids
    Nx, Ny = X.shape;
    Lx, Ly = X[0,-1] - X[0,0], Y[-1,0] - Y[0,0];
    dx, dy, dz = Lx / (Nx - 1), Ly / (Ny - 1), z[1] - z[0];
    idz = 1.0j * dz;
    _idz = - idz;

    # compute k-coordinates
    kx = [np.pi / (Nx * dx) * float(2 * (m - 0.5) - (Nx - 1)) for m in range(Nx)];
    ky = [np.pi / (Ny * dy) * float(2 * (m - 0.5) - (Ny - 1)) for m in range(Ny)];
    Kx, Ky = np.meshgrid(kx, ky);

    # compute general parameters of propagation
    k0 = 2.0 * np.pi / wavelength;
    idz_by_2k = idz / (2.0 * k0 * medium.refIndex);
    # transfer function in free space
    H = fftshift(np.exp(idz_by_2k * (Kx ** 2. + Ky ** 2.)));
    if isinstance(boundary_condition, absorbing_boundary):
        # apply absorbing boundary conditions
        S0 = boundary_condition.apply(X,Y);
    else:
        # input no absorption in computational window
        S0 = np.zeros(U.shape);

    # estimate propagation
    for Z in z:
        # evaluate free space propagation effects in Fourier plane
        U = ifft2(H * fft2(U));
        # evaluate nonlinear and inhomogeneity effects in direct space
        S = S0 + medium.apply_nonlinearity(U);
        for waveguide in medium.waveguides:
            S += waveguide.apply_refractive_index(X,Y,Z);
        U = np.exp(_idz * S) * U;
    return U;
