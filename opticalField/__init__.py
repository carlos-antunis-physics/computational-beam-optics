import numpy as np

def createField(
    F: np.ufunc,
    region: tuple[np.ndarray, np.ndarray],
    center: tuple[float, float] = (0.0, 0.0),
    phase: float | np.ndarray = 0.0,
    incidentWaveK: tuple[float, float, float] = (0.0, 0.0, 0.0),
    normalize: bool = False
) -> np.ndarray:
    """
        createField:
            evaluate optical field in a certain region of transverse plane.

            syntax:
                U = createField(F, region, ...);
            input arguments:
                F: np.ufunc
                    field envelope
                region: tuple[np.ndarray, np.ndarray]
                    X, Y coordinates of points in computational window
                [optional] center: tuple[float, float]
                    x, y coordinates of point in which field are centerd
                [optional] phase: float | np.ndarray
                    phase pattern of field
                [optional] incidentWavevector: tuple[float, float, float]
                    incident wave vector (angles measured in degrees to x and y axis)
                [optional] normalize: bool
                    boolean condition to normalize optical field intensity

    """
    # compute parameters of procedure
    X, Y = region;                  # X, Y of region where field will be evaluated
    xc, yc = center;                # x, y of field origin
    k, phi_x, phi_y = incidentWaveK;# k, angles of incident field
    # optical field envelope evaluation
    U = F(                          # evaluate field translated
        X - xc,
        Y - yc
    ).astype(np.complex128);
    if normalize:
        U = U / abs(U).max().max(); # normalize field intensity
    # multiply wave phasors
    U *= np.exp(1.0j * phase);      # wave phase phasor
    U *= np.exp(-1.0j * k *(        # incidence phasor
        np.tan(phi_x * np.pi / 180.0) * X +
        np.tan(phi_y * np.pi / 180.0) * Y
    ));
    return U;

def intensity(U: np.ndarray) -> np.ndarray:
    return np.abs(U) ** 2.0;

def phase(U: np.ndarray) -> np.ndarray:
    return np.angle(U);