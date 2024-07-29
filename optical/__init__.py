'''
    optical module:
        computational algorithms to estimate classical  optical phenomena
        as light beam propagation through optical media (even those with
        non-linear responses).
'''

__version__ = '1.0.1';                          # optical module version
__name__ = 'optical';                           # module standard name

'''
    importing of useful python packages
'''

import numpy as np
from scipy.constants import degree
from matplotlib import pyplot as plt
from skimage import measure

'''
    wave optics callable utils
'''

# wave number of an optical field
wave_number: callable = lambda wave_length:\
    0. if wave_length == np.inf else 2. * np.pi  / wave_length;

'''
    beam optics callable utils
'''

# Rayleigh range of an optical beam
rayleigh_range: callable = lambda w0, wave_length: w0 ** 2.  * wave_number(wave_length);
# optical beam width along propagation
beam_waist: callable = lambda z, z0, w0:\
    w0 if z0 == 0. else w0 * np.sqrt(1. + (z / z0) ** 2.);
# gaussian beam radius of curvature along propagation
radius_of_curvature: callable = lambda z, z0:\
    np.inf if z == 0. else z * np.sqrt(1. + (z0 / z) ** 2.);
# gaussian beam gouy phase along propagation
gouy_phase: callable = lambda z, z0: np.arctan2(z, z0);

'''
    Dirichlet boundary conditions as optical beam transverse profile
'''

def beam(
    profile: np.ufunc,                          # light beam transverse profile
    region: tuple[np.ndarray, np.ndarray],      # region of transverse profile
    center: tuple[float, float] = (0., 0.),     # cartesian coordinates of beam center
    phase: float | np.ndarray | np.ufunc = 0.,  # phase transverse profile 
) -> np.ndarray:
    '''
        optical.beam
            evaluate the transverse profile of an optical beam along a region on the
            transverse plane.
    '''
    # evaluate coordinates of interest region
    X, Y = region;                              # cartesian coordinates meshgrid
    x0, y0 = center;                            # cartesian coordinates of the center
    x, y = X - x0, Y + y0;
    # evaluate profile along interest region of transverse plane
    try:
        # try to evaluate transverse profile in cartesian coordinates
        psi = profile(x = x, y = y);
    except TypeError:
        # if it raises a TypeError, try to evaluate in polar coordinates
        r, phi = np.sqrt(x**2. + y**2.), np.arctan2(y, x);
        psi = profile(r = r, phi = phi);
    psi[-1:0,:] = 0.; psi[:,-1:0] = 0.;         # ensure zero at boundaries
    # evaluate phasor along interest region of transverse plane
    try:
        # try to evaluate phase transverse profile in cartesian coordinates
        phase = phase(x = x, y = y);
    except TypeError:
        # if it raises a TypeError, try to evaluate in polar coordinates
        r, phi = np.sqrt(x**2. + y**2.), np.arctan2(y, x);
        try:
            phase = phase(r = r, phi = phi);
        except TypeError:
            # if it also raises a TypeError, assume that phase is already evaluated
            if isinstance(phase, np.ndarray):
                # ensure phase perfectly match along the transverse plane meshgird
                if r.shape != phase.shape:
                    raise ValueError('phase must fill precisely the region meshgrid.');
    # ensure that transverse profile was evaluated as a complex function
    return psi.astype(complex) * np.exp(1.j * phase);

# phase due to oblique incidence of an optical beam
def oblique_phasor(
    angulation: tuple[float, float],            # beam oblique angulation
    wave_length: float,                         # wave length of the beam
    region: tuple[np.ndarray, np.ndarray],      # region of transverse profile
    center: tuple[float, float] = (0., 0.),     # cartesian coordinates of beam center
) -> tuple[float, float]:
    '''
        optical.oblique_phasor
            evaluate the wave vector for the given angulations.
    '''
    # evaluate coordinates of interest region
    X, Y = region;                              # cartesian coordinates meshgrid
    x0, y0 = center;                            # cartesian coordinates of the center
    x, y = X - x0, Y + y0;
    # obtain angulation in radians
    ang_x, ang_y = angulation;
    # evaluate the wave vector of the optical beam
    k = wave_number(wave_length);
    k_x, k_y = k * np.tan(ang_x * degree), k * np.tan(ang_y * degree);
    return np.exp(1.j * (k_x * x + k_y * y));

'''
    partial differential equations as optical medium
'''

class waveguide:
    '''
        class optical.waveguide
            initializes an optical waveguide with the most generic properties along z axis.
    '''
    __delta_refractive_index: float;            # main alteration on refractive index
    __alteration: np.ufunc;                     # functional dependence of refractive index
    __center: tuple[np.ufunc, np.ufunc];        # cartesian coordinates of waveguide center
    __is_active: np.ufunc;                      # activeness on a transverse plane
    def __init__(
        self,
        delta_n: float,
        function: np.ufunc = lambda x,y,z: 1.,
        center: tuple[np.ufunc, np.ufunc] = (lambda z: 0., lambda z: 0.),
        zi: float | np.double = -np.inf,
        zf: float | np.double = +np.inf
    ) -> None:
        # evaluate parameters of refractive index alteration on guide
        self.__delta_refractive_index = delta_n;
        self.__alteration = function;
        # evaluate geometric parameters of the optical waveguide
        self.__center = center;
        if (zf > zi):                           # ensure waveguide interval is valid
            if (zi == -np.inf) and (zf == +np.inf):
                # quota being in the real line is a tautology
                self.__is_active = lambda z: True;
            elif (zi != -np.inf):
                # quota only needs to be greater than zi
                self.__is_active = lambda z: (zi <= z);
            elif (zf != +np.inf):
                # quota only needs to be less than zf
                self.__is_active = lambda z: (z <= zf);
            else:
                # quota only needs to be in the range [zi, zf]
                z_m = (zf + zi) / 2.; delta_z = (zf - zi) / 2.;
                self.__is_active = lambda z: np.abs(z - z_m) <= delta_z;
        else:
            raise ValueError('final quota of waveguide must be greater then initial.');
    @property
    def delta_n(self):
        return self.__delta_refractive_index;
    def __call__(self, X: np.ndarray, Y: np.ndarray, z: float | np.ndarray) -> np.ndarray:
        # evaluate coordinates of interest region
        x0, y0 = self.__center[0](z), self.__center[1](z);
        x, y = X - x0, Y - y0;
        try:
            # try to evaluate phase transverse profile in cartesian coordinates
            delta_ref_index = self.delta_n * self.__alteration(x = x, y = y, z = z);
        except TypeError:
            # if it raises a TypeError, try to evaluate in cylindrical coordinates
            r, phi = np.sqrt(x**2. + y**2.), np.arctan2(y, x);
            delta_ref_index = self.delta_n * self.__alteration(r = r, phi = phi, z = z);
        if isinstance(z, float):
            return float(self.__is_active(z)) * delta_ref_index;
        else:
            return np.where(
                self.__is_active(z),
                delta_ref_index,
                0.
            );

class medium:
    '''
        class optical.medium
            initializes an optical medium with waveguides and non-linear responses as space.
    '''
    __refractive_index: float;                  # base refractive index of a medium
    __waveguides = list[waveguide];             # optical waveguides in the medium
    non_linearity: np.ufunc;                    # non-linear response on paraxial equation
    def __init__(
        self,
        n0: float,
        waveguides: list[waveguide] = [],
        non_linearity: np.ufunc | None = None
    ) -> None:
        # evaluate medium base refractive index
        self.__refractive_index = n0;
        # evaluate corrections in optical response
        self.__waveguides = waveguides;         # linear responses
        # non-linear responses
        if non_linearity == None:
            self.non_linearity = lambda psi: np.zeros(psi.shape);
        else:
            self.non_linearity = lambda psi: non_linearity(psi);
    @property
    def n0(self):
        return self.__refractive_index;
    def write(self, waveguide: waveguide) -> None:
        '''
            writes an optical waveguide on the optical medium.
        '''
        self.__waveguides.append(waveguide);
    def __call__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        z: float | np.ndarray
    ) -> np.ndarray:
        '''
            evaluate the linear responses of the medium.
        '''
        return np.zeros_like(X) + sum([wg(X,Y,z) for wg in self.__waveguides]);
    def visualize_waveguides(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        z: np.array,
        cmap = plt.cm.bone
    ) -> None:
        '''
            shows a graphical representation of waveguides written on medium.
        '''
        # creates a three-dimensional figure within the computational window
        fig = plt.figure(); ax = fig.add_subplot(projection = '3d');
        ax.view_init(elev = 30., azim = 80.);   # initialize view angulation
        if self.__waveguides == list():
            # free space has no waveguides to plot
            return;
        # evaluate cartesian coordinates of interest points
        x, y = X[0,:], Y[:,0];
        zVol, xVol, yVol = np.meshgrid(z, y, x);
        # evaluate effective refractive index along computational window
        delta_nVol = self(xVol, yVol, zVol);
        # plot refractive index iso surface
        iso_value = .25 * self.__waveguides[0].delta_n;
        verts, faces, _, _ = measure.marching_cubes(delta_nVol, iso_value);
        ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], cmap = cmap);
        scalar_map = plt.cm.ScalarMappable(cmap = cmap);
        scalar_map.set_array(self.n0 + delta_nVol);
        cbar = fig.colorbar(scalar_map, ax = ax, shrink = 0.5, aspect = 20);
        cbar.set_label('Δn');
        # configure x axis
        ax.set_xlabel('x (μm)');
        ax.set_xticks(np.linspace(0, len(x), 5));
        ax.set_xticklabels([f'{x:.1f}' for x in np.linspace(x[0], x[-1], 5)]);
        # configure y axis
        ax.set_zlabel('y (μm)');
        ax.set_zticks(np.linspace(0, len(y), 5));
        ax.set_zticklabels([f'{y:.1f}' for y in np.linspace(y[0], y[-1], 5)]);
        # configure z axis
        ax.set_ylabel('z (μm)');
        ax.set_yticks(np.linspace(0, len(z), 3));
        ax.set_yticklabels([f'{z:.1f}' for z in np.linspace(z[0], z[-1], 3)]);
