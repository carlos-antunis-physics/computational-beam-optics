'''
    useful packages importing
'''

import numpy as np
from scipy import special as sf

'''
    useful functions from optical module
'''

from .. import Beam as beam
from .. import wave_number
from .. import rayleigh_range as z0
from .. import beam_width as W
from .. import radius_of_curvature as R
from .. import gouy_phase as zeta

'''
    elementary optical beam entry profiles
'''

# gaussian beam entry profile
def gaussian(
    w0: float,                                  # beam width at z = 0 μm
    region: tuple[np.ndarray, np.ndarray],      # transverse plane region to evaluate field
    wave_length: float | None = None,           # wave length of the hermite-gauss beam
    phase: float | np.ndarray | np.ufunc = 0.,  # additional phase of the optical beam
    center: tuple[float, float] = (0., 0.),     # coordinates whereas beam are centered
    z: np.float128 = 0.                         # z coordinate of transverse plane
) -> np.ndarray:
    '''
        optical.beam.gaussian
            evaluate the gaussian beam transverse profile on a defined region of the
            transverse plane.
    '''
    # evaluate parameters of the gaussian beam
    if wave_length == None:
        if z == 0.:
            return beam(
                profile = lambda r, phi: np.exp(-(r/w0)**2.),
                region = region,
                phase = phase,
                center = center
            );
        else:
            raise ValueError(
                'wave length of laser beam are required '
                'to evaluate beam transverse profile'
            );
    _Im_k = -1.j * wave_number(wave_length);    # wave number of the beam
    _z0 = z0(w0, wave_length);                  # rayleigh range of the beam
    _W = W(z, _z0, w0);                         # beam width of the beam
    _2R = 2. * R(z, _z0);                       # twice the radius of curvature of the beam
    _Im_zeta = 1.j * zeta(z, _z0);              # gouy phase of the beam
    _A = (w0 / _W);                             # amplitude of the beam
    # construct entry profile
    return beam(
        profile = lambda r, phi: _A * np.exp(-(r/_W)**2. + _Im_k * (z + r**2./_2R) + _Im_zeta),
        region = region,
        phase = phase,
        center = center
    );

# hermite-gauss beam entry profile
def hermite_gauss(
    w0: float,                                  # beam width at z = 0 μm
    indices: tuple[int, int],                   # indices of hermite modes in x, y coordinates
    region: tuple[np.ndarray, np.ndarray],      # transverse plane region to evaluate field
    wave_length: float | None = None,           # wave length of the hermite-gauss beam
    phase: float | np.ndarray | np.ufunc = 0.,  # additional phase of the optical beam
    center: tuple[float, float] = (0., 0.),     # coordinates whereas beam are centered
    z: np.float128 = 0.                         # z coordinate of transverse plane
) -> np.ndarray:
    '''
        optical.beam.hermite_gauss
            evaluate the hermite-gauss beam transverse profile on a defined region of the
            transverse plane.
    '''
    # construct hermitian modes for x, y axis
    l, m = indices;                             # unpack hermite indices
    Hl, Hm = sf.hermite(l, monic = True), sf.hermite(m, monic = True);
    # evaluate parameters of the hermite-gauss beam
    sq2_by_w0 = np.sqrt(2.) / w0;
    if wave_length == None:
        if z == 0.:
            return beam(
                profile = lambda x, y:  np.exp(-(x**2. + y**2.) / w0 ** 2.) *\
                    Hl(sq2_by_w0 * x) * Hm(sq2_by_w0 * y),
                region = region,
                phase = phase,
                center = center
            );
        else:
            raise ValueError(
                'wave length of laser beam are required '
                'to evaluate beam transverse profile'
            );
    _Im_k = -1.j * wave_number(wave_length);    # wave number of the beam
    _z0 = z0(w0, wave_length);                  # rayleigh range of the beam
    _W = W(z, _z0, w0);                         # beam width of the beam
    _2R = 2. * R(z, _z0);                       # twice the radius of curvature of the beam
    _A = (w0 / _W);                             # amplitude of the beam
    _Im_zeta = 1.j * zeta(z, _z0) * (l + m + 1);# gouy phase of the beam
    # construct entry profile
    return beam(
        profile = lambda x,y: _A * np.exp(_Im_k * (z + (x**2. + y**2.)/_2R) + _Im_zeta) *\
            Hl(sq2_by_w0 * x) * np.exp(-(sq2_by_w0 * x)**2. / 2.) *\
            Hm(sq2_by_w0 * y) * np.exp(-(sq2_by_w0 * y)**2. / 2.),
        region = region,
        phase = phase,
        center = center
    );

# laguerre-gauss beam entry profile
def laguerre_gauss(
    w0: float,                                  # beam width at z = 0 μm
    indices: tuple[int, int],                   # azimuthal and radial indices
    region: tuple[np.ndarray, np.ndarray],      # transverse plane region to evaluate field
    wave_length: float | None = None,           # wave length of the hermite-gauss beam
    phase: float | np.ndarray | np.ufunc = 0.,  # additional phase of the optical beam
    center: tuple[float, float] = (0., 0.),     # coordinates whereas beam are centered
    z: np.float128 = 0.                         # z coordinate of transverse plane
) -> np.ndarray:
    '''
        optical.beam.laguerre_gauss
            evaluate the laguerre-gauss beam transverse profile on a defined region of the
            transverse plane.
    '''
    # construct laguerre mode
    l, m = indices;                             # unpack laguerre indices
    GL = sf.genlaguerre(m, np.abs(l), monic = True);
    # evaluate parameters of the laguerre-gauss beam
    _Im_l = 1.j * l;
    if wave_length == None:
        if z == 0.:
            return beam(
                profile = lambda r, phi: (r / w0) ** np.abs(l) *\
                     GL(2. * (r / w0) ** 2.) * np.exp(-(r / w0) ** 2.) *\
                        np.exp(_Im_l * phi),
                region = region,
                phase = phase,
                center = center
            );
        else:
            raise ValueError(
                'wave length of laser beam are required '
                'to evaluate beam transverse profile'
            );
    _Im_k = -1.j * wave_number(wave_length);    # wave number of the beam
    _z0 = z0(w0, wave_length);                  # rayleigh range of the beam
    _W = W(z, _z0, w0);                         # beam width of the beam
    _2R = 2. * R(z, _z0);                       # twice the radius of curvature of the beam
    _A = (w0 / _W);                             # amplitude of the beam
    _Im_zeta = 1.j * zeta(z, _z0);              # gouy phase of the beam
    _Im_zeta *= float(l + 2 * m + 1);
    # construct entry profile
    return beam(
        profile = lambda r, phi: _A * (r / _W) ** np.abs(l) *\
            GL(2. * (r / _W) ** 2.) * np.exp(-(r / _W) ** 2.) *\
                np.exp(_Im_k * (z + r**2./_2R) + _Im_l * phi + _Im_zeta),
        region = region,
        phase = phase,
        center = center
    );

# bessel beam entry profile
def bessel(
    m: int,                                     # bessel mode index
    kt: float,                                  # bessel mode 'wave number'
    region: tuple[np.ndarray, np.ndarray],      # transverse plane region to evaluate field
    wave_length: float | None = None,           # wave length of the hermite-gauss beam
    phase: float | np.ndarray | np.ufunc = 0.,  # additional phase of the optical beam
    center: tuple[float, float] = (0., 0.),     # coordinates whereas beam are centered
    z: np.float128 = 0.                         # z coordinate of transverse plane
) -> np.ndarray:
    '''
        optical.beam.bessel
            evaluate the bessel beam transverse profile on a defined region of the
            transverse plane.
    '''
    # evaluate parameters of the bessel beam
    _Im_m = 1.j * m;
    if wave_length == None:
        if z == 0.:
            return beam(
                profile = lambda r, phi: sf.jv(m, kt * r) * np.exp(_Im_m * phi),
                region = region,
                phase = phase,
                center = center
            );
        else:
            raise ValueError(
                'wave length of laser beam are required '
                'to evaluate beam transverse profile'
            );
    k = wave_number(wave_length);               # wave number of the beam
    _Im_beta = -1.j * np.sqrt(np.abs(k ** 2. - kt ** 2.));
    # construct entry profile
    return beam(
        profile = lambda r, phi: sf.jv(m, kt * r) * np.exp(_Im_beta * z + _Im_m * phi),
        region = region,
        phase = phase,
        center = center
    );