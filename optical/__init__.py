'''
    optical package:
        computational simulation of light beam propagation through optical media
        with non-linear responses.
'''

__version__ = '1.0.0';
__name__    = 'optical';

'''
    useful packages importing
'''

import numpy as np
from matplotlib import pyplot as plt
from skimage import measure

'''
    optical beam entry profiles (boundary values of optical beam)
'''

# transformation from cartesian coordinates to polar coordinates
_cart2pol: callable = lambda x, y: (np.sqrt(x**2. + y**2.), np.arctan2(y,x));

# wave number of an optical
wave_number: callable = lambda wave_length: 2. * np.pi / wave_length;
# rayleigh range z_0 of a gaussian beam along propagation
rayleigh_range: callable = lambda w0, wave_length: np.pi * w0 ** 2. / wave_length;
# gaussian beam width along propagation
beam_width: callable = lambda z, z0, w0: w0 * np.sqrt(1. + (z / z0) ** 2.);
# gaussian beam radius of curvature  along propagation
radius_of_curvature: callable = lambda z, z0: np.infty if z == 0. else z * (1 + (z0 / z) ** 2.);
# gaussian beam gouy phase along propagation
gouy_phase: callable = lambda z, z0: np.arctan2(z, z0);

# additional phase due to oblique incidence of an optical beam
def incidence_phase(wave_length: float, angulation: tuple[float, float]) -> callable:
    '''
        optical.incidence_phase
            evaluate the additional phase due to oblique incidence of an optical beam.
    '''
    # evaluate the wave number of an optical beam with the specified wave length
    K = wave_number(wave_length);
    # obtain angulation in radians
    ang_x, ang_y = angulation;
    k_x, k_y = K * np.tan(ang_x * np.pi / 180.), K * np.tan(ang_y * np.pi / 180.);
    return lambda x, y: -(k_x * x + k_y * y);

# optical beam construction
def Beam(
    function: np.ufunc,                         # light beam envelope            
    region: tuple[np.ndarray, np.ndarray],      # transverse plane region to evaluate field
    phase: float | np.ndarray | np.ufunc = 0.,  # additional phase of the optical beam
    center: tuple[float, float] = (0., 0.)      # coordinates whereas beam are centered
) -> np.ndarray:
    '''
        optical.Beam
            evaluate the optical beam transverse profile on a defined region of the 
            transverse plane.
    '''
    # evaluate cartesian coordinates of interest points
    X, Y = region;                              # x, y coordinates meshgrid of points on region
    x0, y0 = center;                            # x, y coordinates of point whereas beam are centered
    # evaluate light beam envelope along transverse plane properly
    x, y = X - x0, Y + y0;                      # x, y coordinates in reference frame centered at center
    try:                                        # try to evaluate in cartesian coordinates
        transverse_profile = function(x = x, y = y);
    except TypeError:                           # except if it is described in polar coordinates
        r, phi = _cart2pol(x, y);
        transverse_profile = function(r = r, phi = phi);
    finally:
        # ensure the optical beam is a complex function
        transverse_profile = transverse_profile.astype(np.complex128);
    # evaluate additional phase on the transverse plane properly
    if callable(phase):                         # if phase is a function of coordinates
        try:                                    # try to evaluate it in cartesian coordinates
            phase = phase(x = x, y = y);
        except TypeError:                       # except if it is described in polar coordinates
            r, phi = _cart2pol(x, y);
            phase = phase(r = r, phi = phi);
    return transverse_profile * np.exp(1.j * phase);

# evaluate the optical beam transverse profile in order to make its highest intesity as unitary
normalize: callable = lambda psi: psi / np.abs(psi).max().max();

'''
    optical medium construction (partial diferential equation which describes the light beam propagation)
'''

# optical waveguides construction
class Waveguide:
    '''
        class optical.Waveguide
            creates an optical waveguide with a generic shape along z axis.
    '''
    delta_n0: float;                            # refractive index total variation at optical waveguide
    __gradation: np.ufunc;                      # gradation of refractive index along waveguide
    __shape: np.ufunc;                          # shape variation along z axis
    def __init__(
        self,
        delta_n0: float,                        # refractive index total variation at optical waveguide
        gradation: np.ufunc = lambda x,y,z: 1., # gradation of refractive index along waveguide
        shape: np.ufunc = lambda x,y,z: True    # shape variation along z axis
    ) -> None:
        # evaluate total variation of refractive index due to the optical waveguide
        self.delta_n0 = delta_n0;
        self.__gradation = gradation;           # gradation on refractive index along waveguide
        # obtain the geometrical shape of the optical waveguide
        self.__shape = shape;
    def __call__(self, X: float | np.ndarray, Y: float | np.ndarray, z: float | np.ndarray) -> np.ndarray:
        '''
            evaluate the effective variation of refractive index due to the optical waveguide.
        '''
        # evaluate refractive index variation properly
        try:                                    # try to evaluate in cartesian coordinates
            delta_n0 = self.delta_n0 * self.__gradation(x = X, y = Y, z = z);
        except TypeError:                       # except if it is described in cylindrical coordinates
            R, Phi = _cart2pol(X, Y);
            delta_n0 = self.delta_n0 * self.__gradation(r = R, phi = Phi, z = z);
        # change the refractive index everywhere waveguide shape holds properly
        try:                                    # try to evaluate in cartesian coordinates
            return np.where(
                self.__shape(x = X, y = Y, z = z),
                delta_n0,
                0.
            );
        except TypeError:                       # except if it is described in cylindrical coordinates
            R, Phi = _cart2pol(X, Y);
            return np.where(
                self.__shape(r = R, phi = Phi, z = z),
                delta_n0,
                0.
            );

# optical medium construction
class Medium:
    '''
        class optical.Medium
            creates an optical medium with inhomogeneities due to waveguides and with
            non-linear responses.
    '''
    n0: float;                                  # base refractive index of optical medium
    __waveguides: list[Waveguide];              # optical waveguides written on optical medium
    __non_linear_operator: np.ufunc;            # non-linear response operator
    def __init__(
        self,
        n0: float,                              # base refractive index of optical medium
        waveguides: list[Waveguide] = [],       # optical waveguides written on optical medium
        nonlinearity: np.ufunc = lambda psi: 0. # non-linear response operator
    ) -> None:
        # evaluate base refractive index of optical medium
        self.n0 = n0;
        # evaluate inhomogeneities and non-linear response of the medium
        self.__waveguides = waveguides;
        self.__non_linear_operator = nonlinearity;
    def write(self, waveguide: Waveguide) -> None:
        '''
            write a waveguide on the optical medium.
        '''
        self.__waveguides.append(waveguide);
    def apply_refractive_index(
        self,
        X: float | np.ndarray, Y: float | np.ndarray, z: float | np.ndarray
    ) -> np.ndarray:
        '''
            evaluate the variation on refractive index due to optical waveguides written on optical
            medium.
        '''
        _0 = np.zeros(X.shape);
        return _0 + sum([waveguide(X,Y,z) for waveguide in self.__waveguides]);
    def apply_nonlinearity(self, U: np.ndarray) -> np.ndarray:
        '''
            evaluate the non-linear effects of optical medium on light beam propagation.
        '''
        return self.__non_linear_operator(U);
    def visualize(self, X: np.ndarray, Y: np.ndarray, z: np.ndarray, cmap = plt.cm.gray) -> None:
        '''
            show a graphical representation of the optical waveguides written on the optical medium.
        '''
        # evaluate cartesian coordinates of interest points
        x, y = X[0,:], Y[:,0];                  # x, y coordinates values
        zVol, yVol, xVol = np.meshgrid(z,y,x);  # meshgrid of x, y, z coordinates
        # create a three-dimensional plot
        fig = plt.figure();                     # create a dummy figure
        ax = fig.add_subplot(projection='3d');  # create a frame with 3d view
        ax.view_init(elev = 30., azim = -140.);   
        if self.__waveguides != []:
            # evaluate refractive index along volume
            nVol = self.n0 * np.ones(zVol.shape) + self.apply_refractive_index(xVol, yVol, zVol);
            # get level of waveguide variation
            iso_value = self.n0 + 0.5 * self.__waveguides[0].delta_n0;
            # extract iso surface vertices and faces
            verts, faces, _, _ = measure.marching_cubes(nVol, level = iso_value);
            # plot iso surface
            ax.plot_trisurf(
                verts[:, 0], verts[:, 1], faces, verts[:, 2],
                cmap = cmap,            # colormap
                edgecolor = 'none'
            );
        else:
            # evaluate refractive index along volume
            nVol = self.n0 * np.ones(zVol.shape);
            # set the level of isosurface as the refractive index
            iso_value = 1.5 * self.n0;
        # impress a colorbar
        scalar_map = plt.cm.ScalarMappable(cmap = cmap);
        scalar_map.set_array(nVol - self.n0);
        cbar = fig.colorbar(scalar_map, ax = ax, shrink = 0.5, aspect = 20);
        cbar.set_label('Δn');
        # configure x axis
        ax.set_zlabel('x (μm)');
        ax.set_zticks(np.linspace(0,len(x),5));
        ax.set_zticklabels([f'{x:.1f}' for x in np.linspace(x[0],x[-1],5)]);
        # configure y axis
        ax.set_xlabel('y (μm)');
        ax.set_xticks(np.linspace(0,len(y),5));
        ax.set_xticklabels([f'{x:.1f}' for x in np.linspace(y[0],y[-1],5)]);
        # configure z axis
        ax.set_ylabel('z (μm)');
        ax.set_yticks(np.linspace(0,len(z),3));
        ax.set_yticklabels([f'{x:.1f}' for x in np.linspace(z[0],z[-1],3)]);