import numpy as np

incidence = lambda k, beamAng: dict(
        x = k * np.tan(beamAng[0] * np.pi / 180.0),
        y = k * np.tan(beamAng[1] * np.pi / 180.0)
    );

def create(
    region: tuple[np.ndarray, np.ndarray],
    F: np.ufunc,
    phase: float | np.ndarray[float] = 0.0,
    incidence: dict = incidence(k = 0.0, beamAng = (0.0, 0.0)),
    center: tuple[float, float] = (0.0, 0.0),
    normalize: bool = True
) -> np.ndarray:
    '''
    ## (function) `optical.beam.create`:
        process which evaluate a light beam (`F`) whithin a finite stratum of space (`region`) .

    ### syntax:
            `U = optical.beam.create(F = envelope, region = (X, Y))`
        
    ### parameters:
            `F`: `np.ufunc`
                optical beam envelope universal function dependence in X, Y coordinates.
            `region`: `tuple[np.ndarray, np.ndarray]`
                tuple of X, Y meshgrid where beam will be evaluated.
            [optional] `phase`: `float` : `np.ndarray`
                beam phase value or matrix along the computational region.
            [optional] `incidence`: `dict`
                beam incidente vector dictionary constructed by `optical.beam.incidence`.
            [optional] `center`: `tuple[float, float]`
                tuple of x, y coordinate values to translate beam origin.
            [optional] `normalize`: `bool`
                condition to beam intensity normalization.
    '''
    # compute spatial coordinate parameters
    X, Y = region;                  # x, y axis meshgrid where beam will be evaluated
    xC, yC = center;                # x, y coordinates to translate beam origin
    # optical beam evaluation
    U = F(                          # evaluate beam wave envelope in translated meshgrid
        X - xC,
        Y - yC
    ).astype(np.complex128);
    if normalize:
        U = U / np.abs(U).max().max();
    U *= np.exp(1.0j * phase);      # evaluate phase of the beam
    U *= np.exp(                    # evaluate incidence phasor of the beam
        -1.0j * (incidence['x'] * X + incidence['y'] * Y)
    );
    return U;

def createG(
    region: tuple[np.ndarray, np.ndarray],
    w0: float,
    phase: float | np.ndarray[float] = 0.0,
    incidence: dict = incidence(k = 0.0, beamAng = (0.0, 0.0)),
    center: tuple[float, float] = (0.0, 0.0),
    normalize: bool = True
) -> np.ndarray:
    '''
    ## (function) `optical.beam.createG`:
        process which evaluate a gaussian beam whithin a finite stratum of space (`region`) .

    ### syntax:
            `U = optical.beam.createG(w0 = w0, region = (X, Y))`
        
    ### parameters:
            `w0`: `float`
                value of gaussian beam waist.
            `region`: `tuple[np.ndarray, np.ndarray]`
                tuple of X, Y meshgrid where beam will be evaluated.
            [optional] `phase`: `float` : `np.ndarray`
                beam phase value or matrix along the computational region.
            [optional] `incidence`: `dict`
                beam incidente vector dictionary constructed by `optical.beam.incidence`.
            [optional] `center`: `tuple[float, float]`
                tuple of x, y coordinate values to translate beam origin.
            [optional] `normalize`: `bool`
                condition to beam intensity normalization.
    '''
    return create(
        F = lambda X, Y: np.exp(-(X**2 + Y**2)/w0**2),
        region = region,
        phase = phase,
        incidence = incidence,
        center = center,
        normalize = normalize
    );

from scipy.special import hermite as H
def createHG(
    region: tuple[np.ndarray, np.ndarray],
    w0: float,
    indices: tuple[int, int],
    phase: float | np.ndarray[float] = 0.0,
    incidence: dict = incidence(k = 0.0, beamAng = (0.0, 0.0)),
    center: tuple[float, float] = (0.0, 0.0),
    normalize: bool = True
) -> np.ndarray:
    '''
    ## (function) `optical.beam.createHG`:
        process which evaluate a hermite-gauss beam whithin a finite stratum of space (`region`) .

    ### syntax:
            `U = optical.beam.createHG(w0 = w0, indices = (l,m) , region = (X, Y))`
        
    ### parameters:
            `w0`: `float`
                value of hermite-gauss beam waist.
            `indices`: `tuple[int, int]`
                tuple of x and y hermite polynomial orders.
            `region`: `tuple[np.ndarray, np.ndarray]`
                tuple of X, Y meshgrid where beam will be evaluated.
            [optional] `phase`: `float` : `np.ndarray`
                beam phase value or matrix along the computational region.
            [optional] `incidence`: `dict`
                beam incidente vector dictionary constructed by `optical.beam.incidence`.
            [optional] `center`: `tuple[float, float]`
                tuple of x, y coordinate values to translate beam origin.
            [optional] `normalize`: `bool`
                condition to beam intensity normalization.
    '''
    l, m = indices;                 # HG indices
    _sq2_ov_w0 = np.sqrt(2) / w0;
    _G = lambda i, s: H(i, monic = True)(s) * np.exp(- s**2 / 2.0);
    return create(
        F = lambda X, Y: _G(l, _sq2_ov_w0 * X) * _G(m, _sq2_ov_w0 *Y),
        region = region,
        phase = phase,
        incidence = incidence,
        center = center,
        normalize = normalize
    );

from scipy.special import genlaguerre as L
def createLG(
    region: tuple[np.ndarray, np.ndarray],
    w0: float,
    indices: tuple[int, int],
    phase: float | np.ndarray[float] = 0.0,
    incidence: dict = incidence(k = 0.0, beamAng = (0.0, 0.0)),
    center: tuple[float, float] = (0.0, 0.0),
    normalize: bool = True
) -> np.ndarray:
    '''
    ## (function) `optical.beam.createLG`:
        process which evaluate a laguerre-gauss beam whithin a finite stratum of space (`region`) .

    ### syntax:
            `U = optical.beam.createLG(w0 = w0, indices = (l,m) , region = (X, Y))`
        
    ### parameters:
            `w0`: `float`
                value of laguerre-gauss beam waist.
            `indices`: `tuple[int, int]`
                tuple of azimuthal and radial index of laguerre-gauss beam.
            `region`: `tuple[np.ndarray, np.ndarray]`
                tuple of X, Y meshgrid where beam will be evaluated.
            [optional] `phase`: `float` : `np.ndarray`
                beam phase value or matrix along the computational region.
            [optional] `incidence`: `dict`
                beam incidente vector dictionary constructed by `optical.beam.incidence`.
            [optional] `center`: `tuple[float, float]`
                tuple of x, y coordinate values to translate beam origin.
            [optional] `normalize`: `bool`
                condition to beam intensity normalization.
    '''
    l, m = indices;                 # LG indices
    return create(
        F = lambda X, Y: (np.sqrt(X**2 + Y **2) / w0) ** l * L(l, m, monic = True)(
            2 * (X**2 + Y**2) / w0**2.0
        ) * np.exp(-(X**2 + Y**2)/w0**2),
        region = region,
        phase = phase,
        incidence = incidence,
        center = center,
        normalize = normalize
    );

from scipy.special import jv as J
def createJ(
    region: tuple[np.ndarray, np.ndarray],
    k_t: float,
    m: int,
    phase: float | np.ndarray[float] = 0.0,
    incidence: dict = incidence(k = 0.0, beamAng = (0.0, 0.0)),
    center: tuple[float, float] = (0.0, 0.0),
    normalize: bool = True
) -> np.ndarray:
    '''
    ## (function) `optical.beam.createJ`:
        process which evaluate a bessel beam whithin a finite stratum of space (`region`) .

    ### syntax:
            `U = optical.beam.createJ(k_t = k_t, m = m , region = (X, Y))`
        
    ### parameters:
            `k_t`: `float`
                value k_t parameter of bessel beam.
            `m`: `int`
                order of bessel function.
            `region`: `tuple[np.ndarray, np.ndarray]`
                tuple of X, Y meshgrid where beam will be evaluated.
            [optional] `phase`: `float` : `np.ndarray`
                beam phase value or matrix along the computational region.
            [optional] `incidence`: `dict`
                beam incidente vector dictionary constructed by `optical.beam.incidence`.
            [optional] `center`: `tuple[float, float]`
                tuple of x, y coordinate values to translate beam origin.
            [optional] `normalize`: `bool`
                condition to beam intensity normalization.
    '''
    _J = lambda rho, phi: J(m, k_t * rho) * np.exp(-1.0j * m * phi);
    return create(
        F = lambda X, Y: _J(np.sqrt(X**2 + Y**2), np.arctan2(X, Y)),
        region = region,
        phase = phase,
        incidence = incidence,
        center = center,
        normalize = normalize
    );