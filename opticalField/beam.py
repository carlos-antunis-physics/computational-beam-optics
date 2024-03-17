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
