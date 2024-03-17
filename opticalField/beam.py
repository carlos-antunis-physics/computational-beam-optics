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
from scipy.special import eval_hermite as Hermite
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
    _G = lambda i,S: Hermite(i,S) * np.exp(- S**2 / 2.0);
    return createField(
        F = lambda X, Y: _G(l, _sqrt2 * X / w0) * _G(m, _sqrt2 * Y / w0),
        region = region,
        center = center,
        phase = phase,
        incidentWaveK = incidentWavevector,
        normalize = normalize
    );
