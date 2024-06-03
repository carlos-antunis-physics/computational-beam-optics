import numpy as np
from .. import *
import scipy.special as sf

'''
    beam generical construction and utils.
'''

def create(
    F: np.ufunc,
    region: tuple[np.ndarray, np.ndarray],
    phase: np.float128 | np.ndarray | np.ufunc = 0.0,
    k: wave_vector = wave_vector(k = 0.0),
    center: tuple[np.float128, np.float128] = (0.0, 0.0),
    coordinate_system: coordinate = coordinate.cartesian
) -> np.ndarray:
    '''
    ## `optical.beam.create`
        evaluate a light beam envelope `F` within a rectangular finite
        stratum `region` of the transverse plane.

    ### syntax
        `U = optical.beam.create(F = lambda x,y: A(x,y), region = (X,Y))`
    #### optional parameters
        `phase`: `numpy.float128`, `numpy.ndarray` or `numpy.ufunc`
            beam phase along the simulated `region` on transverse plane.
        `k`:  `optical.wave_vector(k, angulation)`
            wave vector of beam at the transverse plane.
        `center`: `tuple[numpy.float128, numpy.float128]`
            x, y cartesian coordinates of beam center.
        `coordinate_system`: `optical.coordinate`
            coordinate system in which `F` is described instead
            of cartesian coordinates.
    '''
    # compute coordinates
    X0, Y0 = region;                # x, y axis meshgrids
    x0, y0 = center;                # x, y center coordinates
    X, Y = X0 - x0, Y0 - y0;        # x, y axis meshgrids recentered
    match coordinate_system.value:
    # evaluate field in required coordinates
        case coordinate.cartesian.value:
            U = F(X, Y);
            # evaluate beam phase properly
            try:
                Phase = phase(X, Y);
            except TypeError:
                Phase = phase;
        case coordinate.polar.value:
            R, Theta = cart2pol(X, Y);
            U = F(R, Theta);
            # evaluate beam phase properly
            try:
                Phase = phase(R, Theta);
            except TypeError:
                Phase = phase;
    # evaluate field as an complex valued function
    U = U.astype(np.complex128);
    U *= np.exp(                    # insert beam phase
        1.0j * Phase
    );
    U *= np.exp(                    # evaluate beam phase due to incidence
        -1.0j * (k.x * X0 + k.y * Y0)
    );
    return U;

normalize = lambda U: U / np.abs(U).max().max();

'''
    elementary beams construction.
'''

def create_G(
    w0: np.float128,
    region: tuple[np.ndarray, np.ndarray],
    phase: np.float128 | np.ndarray | np.ufunc = 0.0,
    k: wave_vector = wave_vector(k = 0.0),
    center: tuple[np.float128, np.float128] = (0.0, 0.0),
) -> np.ndarray:
    '''
    ## `optical.beam.create_G`
        evaluate a gaussian beam within a rectangular finite stratum
        `region` of the transverse plane.

    ### syntax
        `U = optical.beam.create_G(w0, region = (X,Y))`
    #### optional parameters
        `phase`: `numpy.float128`, `numpy.ndarray` or `numpy.ufunc`
            beam phase along the simulated `region` on transverse plane.
        `k`:  `optical.wave_vector(k, angulation)`
            wave vector of beam at the transverse plane.
        `center`: `tuple[numpy.float128, numpy.float128]`
            x, y cartesian coordinates of beam center.
    '''
    return create(
        F = lambda r, _: np.exp(-(r / w0) ** 2.0),
        region = region,
        phase = phase,
        k = k,
        center = center,
        coordinate_system = coordinate.polar
    );

def create_HG(
    w0: np.float128,
    indices: tuple[int, int],
    region: tuple[np.ndarray, np.ndarray],
    phase: np.float128 | np.ndarray | np.ufunc = 0.0,
    k: wave_vector = wave_vector(k = 0.0),
    center: tuple[np.float128, np.float128] = (0.0, 0.0),
) -> np.ndarray:
    '''
    ## `optical.beam.create_HG`
        evaluate a hermite-gauss beam within a rectangular finite
        stratum `region` of the transverse plane.

    ### syntax
        `U = optical.beam.create_HG(w0, (l,m), region = (X,Y))`
    #### optional parameters
        `phase`: `numpy.float128`, `numpy.ndarray` or `numpy.ufunc`
            beam phase along the simulated `region` on transverse plane.
        `k`:  `optical.wave_vector(k, angulation)`
            wave vector of beam at the transverse plane.
        `center`: `tuple[numpy.float128, numpy.float128]`
            x, y cartesian coordinates of beam center.
    '''
    l, m = indices;
    sq2_by_w0 = np.sqrt(2.0) / w0;
    Gl = lambda x: sf.hermite(l, monic = True)(x) * np.exp(-x ** 2. / 2.); 
    Gm = lambda x: sf.hermite(m, monic = True)(x) * np.exp(-x ** 2. / 2.); 
    return create(
        F = lambda x, y: Gl(sq2_by_w0 * x) * Gm(sq2_by_w0 * y),
        region = region,
        phase = phase,
        k = k,
        center = center,
        coordinate_system = coordinate.cartesian
    );

def create_LG(
    w0: np.float128,
    indices: tuple[int, int],
    region: tuple[np.ndarray, np.ndarray],
    phase: np.float128 | np.ndarray | np.ufunc = 0.0,
    k: wave_vector = wave_vector(k = 0.0),
    center: tuple[np.float128, np.float128] = (0.0, 0.0),
) -> np.ndarray:
    '''
    ## `optical.beam.create_LG`
        evaluate a laguerre-gauss beam within a rectangular finite
        stratum `region` of the transverse plane.

    ### syntax
        `U = optical.beam.create_LG(w0, (rad, azmtl), region = (X,Y))`
    #### optional parameters
        `phase`: `numpy.float128`, `numpy.ndarray` or `numpy.ufunc`
            beam phase along the simulated `region` on transverse plane.
        `k`:  `optical.wave_vector(k, angulation)`
            wave vector of beam at the transverse plane.
        `center`: `tuple[numpy.float128, numpy.float128]`
            x, y cartesian coordinates of beam center.
    '''
    l, m = indices;
    L = lambda r: sf.genlaguerre(m, l, monic = True)(2. * (r / w0) ** 2.);
    rlG = lambda r: (r / w0) ** l * np.exp(-(r / w0) ** 2.);
    return create(
        F = lambda r, theta: L(r) * rlG(r) * np.exp(1.0j * l * theta),
        region = region,
        phase = phase,
        k = k,
        center = center,
        coordinate_system = coordinate.polar
    );

def create_J(
    k_t: np.float128,
    m: int,
    region: tuple[np.ndarray, np.ndarray],
    phase: np.float128 | np.ndarray | np.ufunc = 0.0,
    k: wave_vector = wave_vector(k = 0.0),
    center: tuple[np.float128, np.float128] = (0.0, 0.0),
) -> np.ndarray:
    '''
    ## `optical.beam.create_J`
        evaluate a bessel beam within a rectangular finite
        stratum `region` of the transverse plane.

    ### syntax
        `U = optical.beam.create_J(k_t, m, region = (X,Y))`
    #### optional parameters
        `phase`: `numpy.float128`, `numpy.ndarray` or `numpy.ufunc`
            beam phase along the simulated `region` on transverse plane.
        `k`:  `optical.wave_vector(k, angulation)`
            wave vector of beam at the transverse plane.
        `center`: `tuple[numpy.float128, numpy.float128]`
            x, y cartesian coordinates of beam center.
    '''
    J = lambda r: sf.jv(m, k_t * r);
    return create(
        F = lambda r, theta: J(r) * np.exp(-1.0j * m * theta),
        region = region,
        phase = phase,
        k = k,
        center = center,
        coordinate_system = coordinate.polar
    );