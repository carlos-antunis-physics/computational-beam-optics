'''
    useful python packages
'''

# external python imports
import numpy as np

'''
    generic utils
'''

# cartesian to polar transformation
__cart2pol: np.ufunc = lambda x,y: (np.sqrt(x**2. + y**2.), np.arctan2(y,x));

def _evaluate(
    f:np.ufunc,
    x:np.ndarray,
    y:np.ndarray,
    z: float | np.ndarray = 0.
) -> np.ndarray:
    '''
        optical.utils._evaluate
            evaluate a function in the appropriate coordinate systems.
    '''
    # obtain useful variable names
    var_names = set(f.__code__.co_varnames);
    var_names.discard("_");                                     # avoid dummy variables
    var_names.discard("z");
    # classify coordinate system
    if var_names.issubset({'x','y'}):                           # cartesian coordinates
        match f.__code__.co_argcount:
            case 2:
                return f(x, y);
            case 3:
                return f(x, y, z)
    elif var_names.issubset({'r','phi'}):                       # polar coordinates
        r, phi = __cart2pol(x, y);
        match f.__code__.co_argcount:
            case 2:
                return f(r, phi);
            case 3:
                return f(r, phi, z);
    # if none of the coordinate system have been recognized raise an exception
    raise NotImplementedError(
        f'Not recognized coordinate system in {f.__name__} function: {var_names}'
    );

'''
    optics callable utils
'''

# wave number of an optical beam
wave_number: np.ufunc = lambda wave_length, n0 = 1.:\
    2. * np.pi * n0 / wave_length;
# Rayleigh range of a gaussian beam
rayleigh_range: np.ufunc = lambda w0, wave_length, n0 = 1.:\
    w0 ** 2. * wave_number(wave_length, n0 = n0) / 2.;
# beam width of a gaussian beam
beam_waist: np.ufunc = lambda z, z0, w0:\
    w0 if z == 0. else w0 * np.sqrt(1. + (z / z0) ** 2.);
# radius of curvature of a gaussian beam
radius_of_curvature: np.ufunc = lambda z, z0:\
    np.inf if z == 0. else z * np.sqrt(1. + (z0 / z) ** 2.);
# Gouy phase of an optical beam
gouy_phase: np.ufunc = lambda z, z0:\
    np.arctan2(z, z0);

def oblique_phasor(
    angulation: tuple[float, float],
    wave_lenght: float | np.double,
    region: tuple[np.ndarray, np.ndarray],
    center: tuple[float, float] = (0., 0.)
) -> np.ndarray:
    '''
        optical.utils.oblique_phasor
            evaluate a phasor profile along a region of the transverse plane due
            to an oblique incidence.
    '''
    # evaluate coordinates of interest region
    (X, Y), (x0, y0) = region, center;                          # cartesian coordinates
    x, y = X - x0, Y + y0;                                      # offset coordinates
    # evaluate wavevector due to the oblique incidence
    k, (ang_x, ang_y) = wave_number(wave_lenght), angulation;   # wavevector properties
    k_x, k_y = k * np.tan(np.array([ang_x, ang_y]));
    return np.exp(1.j * (k_x * x + k_y * y));

'''
    experimental utils
'''

def stepwise_phase_otimization(
    input: np.ndarray,
    output: np.ndarray,
    transformation: np.ufunc
) -> np.ndarray:
    '''
        optical.utils.stepwise_phase_otimization
            evaluate the best phase mask by a stepwise genetic algorithm
            presented in (DOI: 10.1016/j.optcom.2008.02.022).
    '''

def continuous_phase_otimization(
    input: np.ndarray,
    output: np.ndarray,
    transformation: np.ufunc
) -> np.ndarray:
    '''
        optical.utils.continuous_phase_otimization
            evaluate the best phase mask by a continuous genetic algorithm
            presented in (DOI: 10.1016/j.optcom.2008.02.022).
    '''

def partitioning_phase_otimization(
    input: np.ndarray,
    output: np.ndarray,
    transformation: np.ufunc
) -> np.ndarray:
    '''
        optical.utils.partitioning_phase_otimization
            evaluate the best phase mask by a partitioning genetic algorithm
            presented in (DOI: 10.1016/j.optcom.2008.02.022).
    '''