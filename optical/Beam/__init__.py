'''
    importing of useful python packages
'''

import numpy as np
from scipy import special as sf

'''
    importing of optical module utils
'''

# import field and phasor constructors
from optical import beam
# import beam optics callable utils
from optical import wave_number
from optical import rayleigh_range, beam_waist, radius_of_curvature, gouy_phase

'''
    simple beam transverse profiles
'''

def gaussian(
    w0: float,                                  # beam waist at zero quota transverse plane
    region: tuple[np.ndarray, np.ndarray],      # region of transverse profile
    center: tuple[float, float] = (0., 0.),     # cartesian coordinates of beam center
    phase: float | np.ndarray | np.ufunc = 0.,  # phase transverse profile 
    wave_length: float | np.double = np.inf,    # wave length of the beam
    z: float = 0.                               # quota coordinate of transverse plane
) -> np.ndarray:
    '''
        optical.Beam.gaussian
            evaluate a gaussian beam transverse profile on a region of transverse plane.
    '''
    if wave_length == np.inf:                   # parse the arguments
        if z != 0.:
            raise ValueError(
                'For a non-zero quota, "wave_lenth" must be informed '
                'in order to avoid abiguities.'
            );
    # evaluate parameters of the gaussian beam transverse profile
    j_k = 1.j * wave_number(wave_length);       # imaginary unit times wave number
    z0 = rayleigh_range(w0, wave_length);       # Rayleigh range of profile
    W = beam_waist(z, z0, w0);                  # beam waist of profile
    _2R = 2. * radius_of_curvature(z, z0);      # twice radius of curvature of profile
    j_zeta = 1.j * gouy_phase(z, z0);           # imaginary unit times Gouy phase
    A = (w0 / W);                               # amplitude of the profile
    WW = W ** 2.;
    return beam(
        profile = lambda r, phi:\
            A * np.exp(-(r**2. / WW)) * np.exp(- j_k * (z + (r**2./_2R)) + j_zeta),
        region = region,
        center = center,
        phase = phase
    );

def hermite_gauss(
    w0: float,                                  # beam waist at zero quota transverse plane
    indices: tuple[int, int],                   # hermite mode indices of cartesian coordinates
    region: tuple[np.ndarray, np.ndarray],      # region of transverse profile
    center: tuple[float, float] = (0., 0.),     # cartesian coordinates of beam center
    phase: float | np.ndarray | np.ufunc = 0.,  # phase transverse profile 
    wave_length: float | np.double = np.inf,    # wave length of the beam
    z: float = 0.                               # quota coordinate of transverse plane
) -> np.ndarray:
    '''
        optical.Beam.hermite_gauss
            evaluate a hermite-gauss beam transverse profile on a region of transverse plane.
    '''
    if wave_length == np.inf:                   # parse the arguments
        if z != 0.:
            raise ValueError(
                'For a non-zero quota, "wave_lenth" must be informed '
                'in order to avoid abiguities.'
            );
    # evaluate parameters of the hermite-gauss beam transverse profile
    j_k = 1.j * wave_number(wave_length);       # imaginary unit times wave number
    z0 = rayleigh_range(w0, wave_length);       # Rayleigh range of profile
    W = beam_waist(z, z0, w0);                  # beam waist of profile
    _2R = 2. * radius_of_curvature(z, z0);      # twice radius of curvature of profile
    j_zeta = 1.j * gouy_phase(z, z0);           # imaginary unit times Gouy phase
    A = (w0 / W);                               # amplitude of the profile
    sqrt2_W = np.sqrt(2.) / W; WW = W ** 2.;
    # obtain modes of hermite polynimials
    lx, ly = indices; j_zeta = float(lx + ly + 1) * j_zeta;
    H_x, H_y = sf.hermite(lx, monic = True), sf.hermite(ly, monic = True);
    return beam(
        profile = lambda x,y:\
            A * np.exp(-(x**2. + y**2.) / WW) * H_x(sqrt2_W * x) * H_y(sqrt2_W * y) * \
                np.exp(-j_k * (z + (x**2. + y**2.)/_2R) + j_zeta),
        region = region,
        center = center,
        phase = phase
    );

def laguerre_gauss(
    w0: float,                                  # beam waist at zero quota transverse plane
    l: int,                                     # azimuthal index of beam
    region: tuple[np.ndarray, np.ndarray],      # region of transverse profile
    m: int = 0,                                 # radial index of beam
    center: tuple[float, float] = (0., 0.),     # cartesian coordinates of beam center
    phase: float | np.ndarray | np.ufunc = 0.,  # phase transverse profile 
    wave_length: float | np.double = np.inf,    # wave length of the beam
    z: float = 0.,                              # quota coordinate of transverse plane
    # geom_phase: float | np.ndarray | np.ufunc = 0.,
) -> np.ndarray:
    '''
        optical.Beam.laguerre_gauss
            evaluate a laguerre-gauss beam transverse profile on a region of transverse
            plane.
    '''
    if wave_length == np.inf:                   # parse the arguments
        if z != 0.:
            raise ValueError(
                'For a non-zero quota, "wave_lenth" must be informed '
                'in order to avoid abiguities.'
            );
    # evaluate parameters of the laguerre-gauss beam transverse profile
    j_k = 1.j * wave_number(wave_length);       # imaginary unit times wave number
    z0 = rayleigh_range(w0, wave_length);       # Rayleigh range of profile
    W = beam_waist(z, z0, w0);                  # beam waist of profile
    _2R = 2. * radius_of_curvature(z, z0);      # twice radius of curvature of profile
    j_zeta = 1.j * gouy_phase(z, z0);           # imaginary unit times Gouy phase
    A = (w0 / W);                               # amplitude of the profile
    j_zeta = float(l + 2*m + 1) * j_zeta;
    WW = W**2.; _2_WW = 2. / WW;
    # obtain modes of hermite polynimials
    _l = np.abs(l); W_l = W ** _l; j_l = 1.j * l;
    L = sf.genlaguerre(m, _l, monic = True);
    return beam(
        profile = lambda r, phi:\
            A * (r ** _l / W_l) * L(_2_WW * r**2.) * np.exp(-(r**2. / WW)) *\
                np.exp(-j_k * (z + (r**2. / _2R)) + j_l * phi + j_zeta),
        region = region,
        center = center,
        phase = phase
    );

def bessel(
    k_t: float,                                 # bessel mode wave number of beam
    m: int,                                     # bessel mode index of beam
    region: tuple[np.ndarray, np.ndarray],      # region of transverse profile
    center: tuple[float, float] = (0., 0.),     # cartesian coordinates of beam center
    phase: float | np.ndarray | np.ufunc = 0.,  # phase transverse profile 
    wave_length: float | np.double = np.inf,    # wave length of the beam
    z: float = 0.                               # quota coordinate of transverse plane
) -> np.ndarray:
    '''
        optical.Beam.bessel
            evaluate a bessel beam transverse profile on a region of transverse plane.
    '''
    if wave_length == np.inf:                   # parse the arguments
        if z != 0.:
            raise ValueError(
                'For a non-zero quota, "wave_lenth" must be informed '
                'in order to avoid abiguities.'
            );
    # evaluate parameters of the bessel beam transverse profile
    k = wave_number(wave_length);               # wave number
    j_beta = 1.j * np.sqrt(np.abs(k**2. - k_t**2.)); j_m = 1.j * m;
    return beam(
        profile = lambda r, phi:\
            sf.jv(m, k_t * r) * np.exp(j_beta * z + j_m * phi),
        region = region,
        center = center,
        phase = phase
    );
