'''
    useful python packages
'''

# external python imports
import numpy as np
from matplotlib import pyplot as plt
from skimage import measure

# interal optical python module imports
from optical import waveguide

'''
    centering methods
'''

def straight(
    zi: float,
    zf: float,
    centers: list[tuple[float, float], tuple[float, float]]
) -> tuple[np.ufunc, np.ufunc]:
    '''
        optical.waveguides.straight
            evaluate a straight line to center a waveguide.
    '''
    # evaluate coordinates of positions
    r_i, r_f = centers;                                         # center positions
    r_i = np.array(r_i); r_f = np.array(r_f);
    dz, delta_r = zf - zi, r_f - r_i;                           # variations along path
    velocity = delta_r / dz;
    return (
        lambda z: r_i[0] + velocity[0] * (z - zi),
        lambda z: r_i[1] + velocity[1] * (z - zi)
    );

def curved(
    zi: float,
    zf: float,
    centers: list[tuple[float, float], tuple[float, float]]
) -> tuple[np.ufunc, np.ufunc]:
    '''
        optical.waveguides.curved
            evaluate a straight line to center a waveguide.
    '''
    # evaluate coordinates of positions
    r_i, r_f = centers;                                         # center positions
    r_i = np.array(r_i); r_f = np.array(r_f);
    dz, delta_r = zf - zi, r_f - r_i;
    d = float(np.sqrt(sum(delta_r ** 2.)));
    # construct center point properly
    if d == 0.:                                                 # center as straight
        return straight(zi, zf, centers);
    else:
        phi = np.arctan2(delta_r[1], delta_r[0]);               # angulation to x-axis
        sin_phi, cos_phi = np.sin(phi), np.cos(phi);
        if phi == 0.:
            R = (dz ** 2. + d ** 2.) / (4. * d);
            ax, bx, sgn_d = r_i[0] + R, r_i[0] + d - R, np.sign(d);
            z_m, half_dz = (zf + zi) / 2., dz / 2.;
            def x(z: float | np.ndarray):
                if isinstance(z, np.ndarray):
                    return np.where(
                        z <= zi + half_dz,
                        np.where(
                            np.abs(z - z_m) < half_dz,
                            -sgn_d * np.sqrt(R ** 2. - (z - zi) ** 2.) + ax,
                            -sgn_d * R + ax
                        ),
                        np.where(
                            np.abs(z - z_m) < half_dz,
                            +sgn_d * np.sqrt(R ** 2. - (z - zf) ** 2.) + bx,
                            +sgn_d * R + bx                    
                        )
                    );
                else:
                    if (z <= zi + half_dz):
                        if (np.abs(z - z_m) < half_dz):
                            return -sgn_d * np.sqrt(R ** 2. - (z - zi) ** 2.) + ax;
                        else:
                            return -sgn_d * R + ax;
                    elif (z > zi + half_dz):
                        if (np.abs(z - z_m) < half_dz):
                            return +sgn_d * np.sqrt(R ** 2. - (z - zf) ** 2.) + bx;
                        else:
                            return +sgn_d * R + bx;
            return (
                lambda z: (cos_phi * (x(z) - r_i[0]) + sin_phi * r_i[1]) + r_i[0],
                lambda z: (sin_phi * (x(z) - r_i[0]) - cos_phi * r_i[1])
            );
        else:
            x0, y0 = curved(zi,zf, centers = [(0., 0.), (d, 0.)]);
            return (
                lambda z: r_i[0] + (cos_phi * x0(z) - sin_phi * y0(z)),
                lambda z: r_i[1] + (sin_phi * x0(z) + cos_phi * y0(z))
            );

'''
    waveguides visualizations
'''

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
    iso_value = .5 * sum([wg.delta_n for wg in waveguides]) / len(waveguides);
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