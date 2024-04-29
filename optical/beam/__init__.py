import numpy as np
from dataclasses import dataclass
from enum import Enum

@dataclass
class coordinate_system (Enum):
    cartesian: bool = True;
    polar: bool = False;

cart2pol = lambda coord: (
        np.sqrt(coord[0] ** 2 + coord[1] ** 2),
        np.arctan2(coord[1], coord[0])
    );
pol2cart = lambda coord: (
        coord[0] * np.cos(coord[1]),
        coord[0] * np.sin(coord[1])
    );

@dataclass
class __incidence_t__:
    x: float;
    y: float;
    def __init__(self, x: float, y: float) -> None:
        self.x = x;
        self.y = y;

incidence = lambda k, beam_angulation: __incidence_t__(
        x = k * np.tan(beam_angulation[0] * np.pi / 180.0),
        y = k * np.tan(beam_angulation[1] * np.pi / 180.0)
    );

def create(
    F: np.ufunc,
    region: tuple[np.ndarray, np.ndarray],
    beam_phase: float | np.ndarray = 0.0,
    incidence = incidence(k = 0.0, beam_angulation = (0.0, 0.0)),
    beam_center: tuple[float, float] = (0.0, 0.0),
    coordinate_system = coordinate_system.cartesian
) -> np.ndarray:
    '''
    ## `optical.beam.create`:
        evaluate a light beam envelope `F` within a finite stratum `region`
        of the transverse plane.

    ### syntax:
        `U = optical.beam.create(F = lambda x,y: envelope(x,y), region = (X, Y))`

    ### parameters:
        `region`: `tuple[numpy.ndarray, numpy.ndarray]`
            meshgrids 2-tuple of transverse plane stratum.
        `F`: `numpy.ufunc`
            optical beam envelope.
        [optional] `phase`: `float` or `np.ndarray`
            beam phase along the `region`.
        [optional] `incidence` = `optical.beam.incidence(k, beam_angulation)`
            beam incidence with `k` as wavenumber and `beam_angulation` as angulation.
        [optional] `beam_center` : `tuple[float, float]`
            x, y cartesian coordinates 2-tuple of beam center.
        [optional] `coordinate_system`
            `coordinate_system.polar` if `F` is described in polar coordinates.
    '''
    # compute spatial coordinates
    X, Y = region;                  # x, y axis meshgrids of transverse plane region
    x0, y0 = beam_center;           # x, y coordinates of beam center
    Xc, Yc = X - x0, Y - y0;        # x, y axis meshgrids recentered
    if not coordinate_system.value:
        Rho, phi = cart2pol(
            coord = (Xc, Yc)
        );
        # evaluate beam
        U = F(Rho, phi);
    else:
        # evaluate beam
        U = F(Xc, Yc);
    U = U.astype(np.complex128);
    U *= np.exp(                    # insert beam phase
        +1.0j * beam_phase
    );
    U *= np.exp(                    # evaluate beam phase due tho incidence
        -1.0j * (
            incidence.x * X + incidence.y * Y
        )
    );
    return U;

normalize = lambda U: U / np.abs(U).max().max();

def create_G(
    w0: float,
    region: tuple[np.ndarray, np.ndarray],
    A: float = 1.0,
    beam_phase: float | np.ndarray = 0.0,
    incidence = incidence(k = 0.0, beam_angulation = (0.0, 0.0)),
    beam_center: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    return create(
        F = lambda r, phi: A * np.exp(-(r / w0) ** 2),
        region = region,
        beam_phase = beam_phase,
        incidence = incidence,
        beam_center = beam_center,
        coordinate_system = coordinate_system.polar
    );

from scipy.special import hermite as H
def create_HG(
    w0: float,
    indices: tuple[int, int],
    region: tuple[np.ndarray, np.ndarray],
    A: float = 1.0,
    beam_phase: float | np.ndarray = 0.0,
    incidence = incidence(k = 0.0, beam_angulation = (0.0, 0.0)),
    beam_center: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    l, m = indices;
    _sq2_ov_w0_ = np.sqrt(2.0) / w0;
    G_l = lambda x: H(l, monic = True)(x) * np.exp(- x ** 2 / 2.0);
    G_m = lambda x: H(m, monic = True)(x) * np.exp(- x ** 2 / 2.0);
    _g = lambda i, s: H(i, monic = True)(s) * np.exp(- s ** 2 / 2.0);
    return create(
        F = lambda x, y: A * G_l(_sq2_ov_w0_ * x) * G_m(_sq2_ov_w0_ * y),
        region = region,
        beam_phase = beam_phase,
        incidence = incidence,
        beam_center = beam_center,
        coordinate_system = coordinate_system.cartesian
    );

from scipy.special import genlaguerre as L
def create_LG(
    w0: float,
    radial_index: int,
    azimuthal_index: int,
    region: tuple[np.ndarray, np.ndarray],
    A: float = 1.0,
    beam_phase: float | np.ndarray = 0.0,
    incidence = incidence(k = 0.0, beam_angulation = (0.0, 0.0)),
    beam_center: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    l, m = radial_index, azimuthal_index;
    L_lm = lambda r: L(m, l, monic = True)(2 * (r / w0) ** 2);
    return create(
        F = lambda r, phi: A * (r / w0) ** l * L_lm(r) * np.exp(-(r / w0) ** 2) * np.exp(1.0j * l * phi),
        region = region,
        beam_phase = beam_phase,
        incidence = incidence,
        beam_center = beam_center,
        coordinate_system = coordinate_system.polar
    );

from scipy.special import jv as J
def create_J(
    k_t: float,
    m: int,
    region: tuple[np.ndarray, np.ndarray],
    A: float = 1.0,
    beam_phase: float | np.ndarray = 0.0,
    incidence = incidence(k = 0.0, beam_angulation = (0.0, 0.0)),
    beam_center: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    return create(
        F = lambda r, phi: A * J(m, k_t * r) * np.exp(-1.0j * m * phi),
        region = region,
        beam_phase = beam_phase,
        incidence = incidence,
        beam_center = beam_center,
        coordinate_system = coordinate_system.polar
    );