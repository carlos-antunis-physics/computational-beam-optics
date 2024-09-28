'''
    useful python packages
'''

# external python imports
import numpy as np
from scipy import special as sf

# interal optical python module imports
from optical import Beam
from optical.utils import wave_number
from optical.utils import rayleigh_range, beam_waist, radius_of_curvature, gouy_phase

'''
    cylindrical symetrical transverse profiles
'''

def gaussian(
    w0: float,
    region: tuple[np.ndarray, np.ndarray],
    center: tuple[float, float] = (0., 0.),
    phase: float | np.ndarray | np.ufunc = 0.,
    z: float = 0.,
    wave_length: float | np.double = np.inf
) -> np.ndarray:
    '''
        optical.beam.gaussian
            evaluate the transverse profile of a gaussian beam.
    '''
    if wave_length == np.inf:                   # parse the arguments
        if z != 0.:
            raise ValueError(
                'For a non-zero quota, "wave_length" must be informed '
                'in order to avoid abiguities.'
            );
    # evaluate parameters of the gaussian beam transverse profile
    j_k = 1.j * wave_number(wave_length);
    z0 = rayleigh_range(w0, wave_length);
    W = beam_waist(z, z0, w0);
    _2R = 2. * radius_of_curvature(z, z0);
    j_zeta = 1.j * gouy_phase(z, z0);
    A = (w0 / W);
    WW = W ** 2.;
    return Beam(
        function = lambda r, _:\
            A * np.exp(-(r**2. / WW)) * np.exp(- j_k * (z + (r**2./_2R)) + j_zeta),
        region = region,
        center = center,
        phase = phase
    );

def laguerre_gauss(
    w0: float,
    l: int,
    region: tuple[np.ndarray, np.ndarray],
    center: tuple[float, float] = (0., 0.),
    m: int = 0,
    phase: float | np.ndarray | np.ufunc = 0.,
    z: float = 0.,
    wave_length: float | np.double = np.inf
) -> np.ndarray:
    '''
        optical.beam.laguerre_gauss
            evaluate the transverse profile of a laguerre-gauss beam.
    '''
    if wave_length == np.inf:                   # parse the arguments
        if z != 0.:
            raise ValueError(
                'For a non-zero quota, "wave_length" must be informed '
                'in order to avoid abiguities.'
            );
    # evaluate parameters of the laguerre-gauss beam transverse profile
    j_k = 1.j * wave_number(wave_length);
    z0 = rayleigh_range(w0, wave_length);
    W = beam_waist(z, z0, w0);
    _2R = 2. * radius_of_curvature(z, z0);
    j_zeta = 1.j * gouy_phase(z, z0);
    A = (w0 / W);
    j_zeta = float(l + 2*m + 1) * j_zeta;
    WW = W**2.; _2_WW = 2. / WW;
    # obtain modes of hermite polynimials
    _l = np.abs(l); W_l = W ** _l; j_l = 1.j * l;
    L = sf.genlaguerre(m, _l, monic = True);
    return Beam(
        function = lambda r, phi:\
            A * (r ** _l / W_l) * L(_2_WW * r**2.) * np.exp(-(r**2. / WW)) *\
                np.exp(-j_k * (z + (r**2. / _2R)) + j_l * phi + j_zeta),
        region = region,
        center = center,
        phase = phase
    );


def bessel(
    k_t: float,
    m: int,
    region: tuple[np.ndarray, np.ndarray],
    center: tuple[float, float] = (0., 0.),
    phase: float | np.ndarray | np.ufunc = 0.,
    z: float = 0.,
    wave_length: float | np.double = np.inf
) -> np.ndarray:
    '''
        optical.beam.bessel
            evaluate the transverse profile of a bessel beam.
    '''
    if wave_length == np.inf:                   # parse the arguments
        if z != 0.:
            raise ValueError(
                'For a non-zero quota, "wave_length" must be informed '
                'in order to avoid abiguities.'
            );
    # evaluate parameters of the bessel beam transverse profile
    k = wave_number(wave_length);               # wave number
    j_beta = 1.j * np.sqrt(np.abs(k**2. - k_t**2.)); j_m = 1.j * m;
    return Beam(
        function = lambda r, phi:\
            sf.jv(m, k_t * r) * np.exp(j_beta * z + j_m * phi),
        region = region,
        center = center,
        phase = phase
    );

'''
    rectangular symetrical transverse profiles
'''

def hermite_gauss(
    w0: float,
    indices: tuple[int, int],
    region: tuple[np.ndarray, np.ndarray],
    center: tuple[float, float] = (0., 0.),
    phase: float | np.ndarray | np.ufunc = 0.,
    z: float = 0.,
    wave_length: float | np.double = np.inf
) -> np.ndarray:
    '''
        optical.beam.hermite_gauss
            evaluate the transverse profile of a hermite-gauss beam.
    '''
    if wave_length == np.inf:                   # parse the arguments
        if z != 0.:
            raise ValueError(
                'For a non-zero quota, "wave_length" must be informed '
                'in order to avoid abiguities.'
            );
    # evaluate parameters of the hermite-gauss beam transverse profile
    j_k = 1.j * wave_number(wave_length);
    z0 = rayleigh_range(w0, wave_length);
    W = beam_waist(z, z0, w0);
    _2R = 2. * radius_of_curvature(z, z0);
    j_zeta = 1.j * gouy_phase(z, z0);
    A = (w0 / W);
    sqrt2_W = np.sqrt(2.) / W; WW = W ** 2.;
    # obtain modes of hermite polynimials
    lx, ly = indices; j_zeta = float(lx + ly + 1) * j_zeta;
    H_x, H_y = sf.hermite(lx, monic = True), sf.hermite(ly, monic = True);
    return Beam(
        function = lambda x, y:\
            A * np.exp(-(x**2. + y**2.) / WW) * H_x(sqrt2_W * x) * H_y(sqrt2_W * y) * \
                np.exp(-j_k * (z + (x**2. + y**2.)/_2R) + j_zeta),
        region = region,
        center = center,
        phase = phase
    );