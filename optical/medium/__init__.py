import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''
    waveguide construction.
'''

class waveguide:
    def __init__(
        self,
        geometry: np.ufunc,
        delta_n: np.float128 | np.complex128 | np.ufunc,
        zi: np.float128 = 0.0,
        zf: np.float128 = +np.infty,
        color: str = 'black'
    ) -> None:
        '''
        ## `optical.medium.waveguide`
            creates an waveguide with `geometry` and an extension of (`zi`, `zf`),
            furthermore `delta_n` as refractive index alteration.
    
        ### syntax
            `WG = optical.medium.waveguide(
                geometry = lambda x,y,z: isInWG(x,y,z),
                delta_n = refIndex_variation
            )`
            `WG = optical.medium.waveguide(
                geometry = lambda x,y,z: isInWG(x,y,z),
                delta_n = lambda x,y,z: refIndex_variation(x,y,z)
            )`
        #### optional parameters
            `zi`: `numpy.float128`
                z coordinate where guide starts. 
            `zf`: `numpy.float128`
                z coordinate where guide ends. 
            `color`: `str`
                color to represent the waveguide in visualizations.
        '''
        # impose guide_geometry as condition to be in waveguide
        if zi == 0.0 and zf == +np.infty:
            self.is_in = lambda x,y,z: geometry(x,y,z);
        elif zi != 0.0:
            self.is_in = lambda x,y,z: (zi <= z) & geometry(x,y,z);
        elif zf != +np.infty:
            self.is_in = lambda x,y,z: geometry(x,y,z) & (z <= zf);
        else: # both diferent
            self.is_in = lambda x,y,z: (zi <= z) & geometry(x,y,z) & (z <= zf);
        # impose waveguide refractive index variation as lambda
        self.__deltan = delta_n;
        # set the waveguide representative color
        self.color = color;
    def apply_refractive_index(
        self,
        X: np.ndarray | np.float128,
        Y: np.ndarray | np.float128,
        z: np.ndarray | np.float128
    ) -> np.ndarray:
        '''
        ## `[optical.medium.waveguide] WG.apply_refractive_index`
            obtain the index alteration in (`X`,`Y`,`z`) due to the
            waveguide.
    
        ### syntax
            `WG.apply_refractive_index(X,Y,z)`
        '''
        if callable(self.__deltan):
            __deltan = self.__deltan(X,Y,z).astype(np.complex128);
        else:
            __deltan = self.__deltan;
        # change refractive index only if is in waveguide
        dn = np.where(
            self.is_in(X,Y,z),
            __deltan,
            0.0
        );
        return dn;

def visualize(
    waveguides: list[waveguide],
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray
) -> None:
    '''
        ## `optical.medium.visualize`
            show an graphical representation of a waveguide list.
    
        ### syntax
            `optical.medium.visualize([WG], colors, x, y, z)`
    '''
    # create spatial meshgrid
    X, Z, Y = np.meshgrid(x,z,y);

    # create voxel properties
    WG = np.zeros(Z.shape, dtype = 'bool');
    Colors = np.empty(WG.shape, dtype = object);

    # create truth matrix along voxels
    F = np.zeros(WG.shape, dtype = 'bool');
    T = np.ones(WG.shape, dtype = 'bool');

    #initialize waveguide volume
    wg: np.ndarray = np.zeros((0,0), dtype = 'bool');

    # for each waveguide
    iWG = 0;
    for waveguide in waveguides:
        # evaluate where voxel are filled due to it
        wg = np.where(
            waveguide.is_in(X,Y,Z),
            T, F
        );
        WG = WG | wg;
        
        # impose its color
        Colors[wg] = waveguide.color;

        iWG += 1;

    #initialize 3d plot
    fig = plt.figure();
    ax = plt.axes(projection = '3d');

    # configure z axis
    ax.set_xlabel('z (μm)');

    __loc = np.linspace(0,len(z),5);
    __lab = np.linspace(z[0],z[-1],5);
    __labels = [f'{x:.1f}' for x in __lab];

    ax.set_xticks(__loc);
    ax.set_xticklabels(__labels);

    # configure x axis
    ax.set_ylabel('x (μm)');

    __loc = np.linspace(0,len(x),5);
    __lab = np.linspace(x[0],x[-1],5);
    __labels = [f'{x:.1f}' for x in __lab];

    ax.set_yticks(__loc);
    ax.set_yticklabels(__labels);

    # configure y axis
    ax.set_zlabel('y (μm)');

    __loc = np.linspace(0,len(y),5);
    __lab = np.linspace(y[0],y[-1],5);
    __labels = [f'{x:.1f}' for x in __lab];

    ax.set_zticks(__loc);
    ax.set_zticklabels(__labels);

    # plot voxels
    ax.voxels(WG, facecolors = Colors);

'''
    waveguide geometry constructors.
'''

def rectangular(
    lengths: tuple[np.float128, np.float128] | tuple[np.ufunc, np.ufunc],
    center: tuple[np.float128, np.float128] | tuple[np.ufunc, np.ufunc] = (0.0, 0.0)
) -> np.ufunc:
    '''
        ## `optical.medium.rectangular`
            constructs a rectangular cuboid geometry with refered `lengths`.
    
        ### syntax
            `optical.medium.rectangular(lengths = (Lx, Ly))`
            `optical.medium.rectangular(lengths = (Lx(z), Ly(z)))`
        #### optional parameters
            `center`: `tuple[numpy.float128, numpy.float128] | tuple[np.ufunc, np.ufunc]`
                cartesian coordinates in where rectangles are centered. 
    '''
    # evaluate geometry parameters
    x0, y0 = center;
    Lx, Ly = lengths;

    # evaluate geometry by callability of parameters
    if callable(x0) and callable(y0):
        # center moving along z axis
        if callable(Lx) and callable(Ly):
            return lambda x,y,z: (np.abs(x - x0(z)) <= (Lx(z) / 2.)) & (np.abs(y - y0(z)) <= (Ly(z) / 2.));
        elif callable(Lx):
            return lambda x,y,z: (np.abs(x - x0(z)) <= (Lx(z) / 2.)) & (np.abs(y - y0(z)) <= (Ly / 2.));
        elif callable(Ly):
            return lambda x,y,z: (np.abs(x - x0(z)) <= (Lx / 2.)) & (np.abs(y - y0(z)) <= (Ly(z) / 2.));
        else:
            return lambda x,y,z: (np.abs(x - x0(z)) <= (Lx / 2.)) & (np.abs(y - y0(z)) <= (Ly / 2.));
    elif callable(x0):
        # center moving in x axis along z axis
        if callable(Lx) and callable(Ly):
            return lambda x,y,z: (np.abs(x - x0(z)) <= (Lx(z) / 2.)) & (np.abs(y - y0) <= (Ly(z) / 2.));
        elif callable(Lx):
            return lambda x,y,z: (np.abs(x - x0(z)) <= (Lx(z) / 2.)) & (np.abs(y - y0) <= (Ly / 2.));
        elif callable(Ly):
            return lambda x,y,z: (np.abs(x - x0(z)) <= (Lx / 2.)) & (np.abs(y - y0) <= (Ly(z) / 2.));
        else:
            return lambda x,y,z: (np.abs(x - x0(z)) <= (Lx / 2.)) & (np.abs(y - y0) <= (Ly / 2.));
    elif callable(y0):
        # center moving in y axis along z axis
        if callable(Lx) and callable(Ly):
            return lambda x,y,z: (np.abs(x - x0) <= (Lx(z) / 2.)) & (np.abs(y - y0(z)) <= (Ly(z) / 2.));
        elif callable(Lx):
            return lambda x,y,z: (np.abs(x - x0) <= (Lx(z) / 2.)) & (np.abs(y - y0(z)) <= (Ly / 2.));
        elif callable(Ly):
            return lambda x,y,z: (np.abs(x - x0) <= (Lx / 2.)) & (np.abs(y - y0(z)) <= (Ly(z) / 2.));
        else:
            return lambda x,y,z: (np.abs(x - x0) <= (Lx / 2.)) & (np.abs(y - y0(z)) <= (Ly / 2.));
    else:
        # center fixed along z axis
        if callable(Lx) and callable(Ly):
            return lambda x,y,z: (np.abs(x - x0) <= (Lx(z) / 2.)) & (np.abs(y - y0) <= (Ly(z) / 2.));
        elif callable(Lx):
            return lambda x,y,z: (np.abs(x - x0) <= (Lx(z) / 2.)) & (np.abs(y - y0) <= (Ly / 2.));
        elif callable(Ly):
            return lambda x,y,z: (np.abs(x - x0) <= (Lx / 2.)) & (np.abs(y - y0) <= Ly(z) / 2.);
        else:
            return lambda x,y,z: (np.abs(x - x0) <= (Lx / 2.)) & (np.abs(y - y0) <= (Ly / 2.));

def circular(
    radius: np.float128 | np.ufunc,
    center: tuple[np.float128, np.float128] | tuple[np.ufunc, np.ufunc] = (0.0, 0.0)
) -> np.ufunc:
    '''
        ## `optical.medium.circular`
            constructs a circular cylinder geometry with refered `radius`.
    
        ### syntax
            `optical.medium.circular(radius = R)`
            `optical.medium.circular(radius = R(z))`
        #### optional parameters
            `center`: `tuple[numpy.float128, numpy.float128] | tuple[np.ufunc, np.ufunc]`
                cartesian coordinates in where circles are centered.
    '''
    # evaluate geometry parameters
    x0, y0 = center;

    # evaluate geometry by callability of parameters
    if callable(x0) and callable(y0):
        # center moving along z axis
        if callable(radius):
            return lambda x,y,z: ((x - x0(z))**2. + (y - y0(z))**2. <= radius(z) ** 2.);
        else:
            return lambda x,y,z: ((x - x0(z))**2. + (y - y0(z))**2. <= radius ** 2.);
    elif callable(x0):
        # center moving in x axis along z axis
        if callable(radius):
            return lambda x,y,z: ((x - x0(z))**2. + (y - y0)**2. <= radius(z) ** 2.);
        else:
            return lambda x,y,z: ((x - x0(z))**2. + (y - y0)**2. <= radius ** 2.);
    elif callable(y0):
        # center moving in y axis along z axis
        if callable(radius):
            return lambda x,y,z: ((x - x0)**2. + (y - y0(z))**2. <= radius(z) ** 2.);
        else:
            return lambda x,y,z: ((x - x0)**2. + (y - y0(z))**2. <= radius ** 2.);
    else:
        # center fixed along z axis
        if callable(radius):
            return lambda x,y,z: ((x - x0)**2. + (y - y0)**2. <= radius(z) ** 2.);
        else:
            return lambda x,y,z: ((x - x0)**2. + (y - y0)**2. <= radius ** 2.);
