'''
    optical python module:
        computational algorithms to estimate light beam propagation through optical
        media, even the non-linear ones.
    
    author:
        Carlos Antunis Bonfim da Silva Santos

'''

__version__ = "2.0.0";                                          # optical module version

'''
    useful python packages
'''

# external python imports
import numpy as np
from matplotlib import pyplot as plt
from skimage import measure

# interal optical python module imports
from .utils import _evaluate

'''
    Dirichlet boundary conditions definition by an transverse profile of an optical beam
'''

def Beam(
    function: np.ufunc,
    region: tuple[np.ndarray, np.ndarray],
    center: tuple[float, float] = (0., 0.),
    phase: float | np.ndarray | np.ufunc = 0.
) -> np.ndarray:
    '''
        optical.Beam
            evaluate the transverse profile of an optical beam along a region of
            the transverse plane.
    '''
    # evaluate coordinates of interest region
    (X, Y), (x0, y0) = region, center;                          # cartesian coordinates
    x, y = X - x0, Y + y0;                                      # offset coordinates
    # obtain transverse profile and phase in apropriate coordinate systems
    function.__name__ = 'transverse profile';                   # error handling
    field = _evaluate(function, x, y);
    if callable(phase):
        phase.__name__ = 'additional phase profile';            # error handling
        phase = _evaluate(phase, x, y);
    return field.astype(complex) * np.exp(1.j * phase);
 
'''
    optical inhomogeneities definition as optical waveguides
'''

class waveguide:
    '''
        class optical.waveguide
    '''
    __delta_n0: float;
    __alteration: np.ufunc;
    __center: tuple[np.ufunc, np.ufunc];
    __is_active: np.ufunc;
    def __init__(
        self,
        delta_n: float,
        function: np.ufunc,
        center: tuple[np.ufunc, np.ufunc] = (lambda z: 0., lambda z: 0.),
        zi: float | np.double = -np.inf,
        zf: float | np.double = +np.inf,
    ) -> None:
        # define refractive index alteration parameters
        self.__delta_n0 = delta_n;
        self.__alteration = function;
        # define geometric parameters
        self.__center = center;
        if zf < zi:
            raise ValueError("Final quota of waveguide is less then inital.");
        else:
            if (zi == -np.inf) and (zf == +np.inf):
                # quota being in the real line is a tautology
                self.__is_active = lambda z: True;
            elif (zi != -np.inf):
                # quota must be greater then zi
                self.__is_active = lambda z: (zi <= z);
            elif (zf != +np.inf):
                # quota must be less then zf
                self.__is_active = lambda z: (z <= zf);
            else:
                # quota must be closer from the mean that half of a range
                z_m = (zf + zi) / 2.;
                half_range_z = (zf - zi) / 2.;
                self.__is_active = lambda z: np.abs(z - z_m) <= half_range_z;
    @property
    def delta_n(self):
        return self.__delta_n0;
    def __call__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        z: float | np.ndarray
    ) -> np.ndarray:
        # evaluate coordinates of interest region
        x0, y0 = self.__center[0](z), self.__center[1](z);      # guide center
        x, y = X - x0, Y + y0;                                  # offset coordinates
        # obtain transverse profile and phase in apropriate coordinate systems
        self.__alteration.__name__ = 'refractive index';        # error handling
        delta_ref_index = self.delta_n * _evaluate(self.__alteration, x, y, z);
        if isinstance(z, np.ndarray):
            return np.where(self.__is_active(z), delta_ref_index, 0.);
        return float(self.__is_active(z)) * delta_ref_index;

def visualize(
    waveguides: list[waveguide],
    X: np.ndarray, Y: np.ndarray, z: np.array,
    fig: plt.Figure,
    cmap = plt.cm.bone
) -> None:
    '''
        optical.waveguides.visualize
            shows a graphical representation of a waveguide list.
    '''
    # create a three-dimensional figure within the computational window
    ax = fig.add_subplot(projection = '3d');
    ax.view_init(elev = 30., azim = 80.);                       # view angulation
    # free space has no waveguides to show
    if waveguides == list():
        return;
    # evaluate cartesian coordinates of interest points
    x, y = X[0,:], Y[:,0];
    z_vol, y_vol, x_vol = np.meshgrid(z, y, x);
    # evaluate refractive index alteration due to inhomogeneities
    n_vol = np.zeros_like(z_vol) + sum([wg(x_vol, y_vol, z_vol) for wg in waveguides]);
    # compute refractive index alteration iso surface
    iso_value = sum([wg.delta_n for wg in waveguides]) / len(waveguides);
    verts, faces, _, _ = measure.marching_cubes(n_vol, iso_value);
    # plot isosurface as a triangular surface
    ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], cmap = cmap);
    # insert colorbar
    sm = plt.cm.ScalarMappable(cmap = cmap);
    sm.set_array(n_vol);
    cbar = fig.colorbar(sm, ax = ax, shrink = .5, aspect = 20);
    cbar.set_label('Δn');
    # configure x axis
    ax.set_zlabel('x (μm)');
    ax.set_zticks(np.linspace(0, len(x), 5));
    ax.set_zticklabels([f'{x:.1f}' for x in np.linspace(x[0], x[-1], 5)]);
    # configure y axis
    ax.set_xlabel('y (μm)');
    ax.set_xticks(np.linspace(0, len(y), 5));
    ax.set_xticklabels([f'{y:.1f}' for y in np.linspace(y[-1], y[0], 5)]);
    # configure z axis
    ax.set_ylabel('z (μm)');
    ax.set_yticks(np.linspace(0, len(z), 3));
    ax.set_yticklabels([f'{z:.1f}' for z in np.linspace(z[0], z[-1], 3)]);

'''
    partial differential equation definition as an optical medium
'''

class medium:
    '''
        class optical.medium
    '''
    __refractive_index: float;
    __waveguides: list[waveguide];
    non_linearity: np.ufunc;
    def __init__(
        self,
        n0: float,
        waveguides: list[waveguide] = list(),
        non_linearity: np.ufunc | None = None
    ) -> None:
        # define first order effects on propagation
        self.__refractive_index = n0;
        # define inhomogeneity effects on propagation
        self.__waveguides = waveguides;
        # define higher order optical responses on propagation
        if non_linearity is None:
            self.non_linearity = lambda psi: np.zeros_like(psi, dtype = float);
        else:
            self.non_linearity = non_linearity;
    @property
    def n0(self):
        return self.__refractive_index;
    def write_waveguide(
        self,
        waveguide: waveguide
    ) -> None:
        self.__waveguides.append(waveguide);
    def visualize_waveguides(
        self,
        X: np.ndarray, Y: np.ndarray, z: np.array,
        fig: plt.Figure,
        cmap = plt.cm.bone
    ) -> None:
        visualize(self.__waveguides,X,Y,z,fig,cmap);
    def inhomogeneity(
        self,
        X: np.ndarray, Y: np.ndarray, z: float | np.ndarray
    ) -> None:
        if isinstance(z, float):
            delta_n = np.zeros((X.shape[0], Y.shape[1]));
        else:
            delta_n = np.zeros_like(z);
        return delta_n + sum([wg(X,Y,z) for wg in self.__waveguides]);
