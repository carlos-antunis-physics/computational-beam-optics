import numpy as np
from opticalField import createField

# gaussian beam constructor
def Gaussian(
    w0 : float,
    region: tuple[np.ndarray, np.ndarray],
    center: tuple[float, float] = (0.0, 0.0),
    phase: float | np.ndarray = 0.0,
    incidentWavevector: tuple[float, float, float] = (0.0, 0.0, 0.0),
    normalize: bool = False
) -> np.ndarray:
    return createField(
        F = lambda X, Y: np.exp(-(X**2 + Y**2)/w0**2),
        region = region,
        center = center,
        phase = phase,
        incidentWaveK = incidentWavevector,
        normalize = normalize
    );

# hermite-gauss beam constructor
from scipy.special import eval_hermite as hermite
def HermiteGauss(
    index: tuple[int, int],
    w0: float,
    region: tuple[np.ndarray, np.ndarray],
    center: tuple[float, float] = (0.0, 0.0),
    phase: float | np.ndarray = 0.0,
    incidentWavevector: tuple[float, float, float] = (0.0, 0.0, 0.0),
    normalize: bool = False
) -> np.ndarray:
    l, m = index;
    _sqrt2 = np.sqrt(2);
    _G = lambda i,S: hermite(i,S) * np.exp(- S**2 / 2.0);
    return createField(
        F = lambda X, Y: _G(l, _sqrt2 * X / w0) * _G(m, _sqrt2 * Y / w0),
        region = region,
        center = center,
        phase = phase,
        incidentWaveK = incidentWavevector,
        normalize = normalize
    );

# laguerre-gauss beam constructor
from scipy.special import assoc_laguerre as laguerre
def LaguerreGauss(
    index: tuple[int, int],
    w0: float,
    region: tuple[np.ndarray, np.ndarray],
    center: tuple[float, float] = (0.0, 0.0),
    phase: float | np.ndarray = 0.0,
    incidentWavevector: tuple[float, float, float] = (0.0, 0.0, 0.0),
    normalize: bool = False
) -> np.ndarray:
    l, m = index;
    return createField(
        F = lambda X, Y: (np.sqrt(X**2 + Y **2) / w0) ** l * laguerre(
            2 * (X**2 + Y**2) / w0 ** 2, l, m
        ) * np.exp(-(X**2 + Y**2)/w0**2),
        region = region,
        center = center,
        phase = phase,
        incidentWaveK = incidentWavevector,
        normalize = normalize
    );

# bessel beam constructor
from scipy.special import jv as bessel
def Bessel(
    m: int,
    k_t: float,
    region: tuple[np.ndarray, np.ndarray],
    center: tuple[float, float] = (0.0, 0.0),
    phase: float | np.ndarray = 0.0,
    incidentWavevector: tuple[float, float, float] = (0.0, 0.0, 0.0),
    normalize: bool = False
) -> np.ndarray:
    _J = lambda rho, theta: bessel(m, k_t * rho) * np.exp(-1.0j * m * theta);
    return createField(
        F = lambda X, Y: _J(np.sqrt(X**2 + Y**2), np.arctan2(X, Y)),
        region = region,
        center = center,
        phase = phase,
        incidentWaveK = incidentWavevector,
        normalize = normalize
    );