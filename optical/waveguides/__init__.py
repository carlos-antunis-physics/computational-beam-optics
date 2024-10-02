'''
    useful python packages
'''

# external python imports
import numpy as np

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
    simple optical waveguides shapes
'''

def rectangular(
    delta_n: float,
    lengths: tuple[float | np.ufunc, float | np.ufunc],
    zi: float,
    zf: float,
    curve: tuple[np.ufunc, np.ufunc] = straight,
    angulation: float = 0.,
    positions: list[tuple[float, float], tuple[float, float]] = [(0.,0.),(0.,0.)]
) -> waveguide:
    '''
        optical.waveguide.rectangular
          initializes an optical waveguide with rectangular shape.
    '''
    lx, ly = lengths;                                           # guide lengths of each coordinate
    Lx = lx if callable(lx) else lambda z: lx;
    Ly = ly if callable(ly) else lambda z: ly;
    rect = lambda u, L: (np.abs(u) < 0.5 * L);
    cos_phi, sin_phi = np.cos(angulation), np.sin(angulation);
    return waveguide(
        delta_n = delta_n,
        function = lambda x, y, z: rect(
            cos_phi * x - sin_phi * y, Lx(z)
        ) * rect(
            sin_phi * x + cos_phi * y, Ly(z)
        ),
        center = curve(zi, zf, positions),
        zi = zi,
        zf = zf
    );

def twisted_rectangular(
    delta_n: float,
    lengths: tuple[float | np.ufunc, float | np.ufunc],
    zi: float,
    zf: float,
    curve: tuple[np.ufunc, np.ufunc] = straight,
    angulations: tuple[float, float] = (0., 0.),
    positions: list[tuple[float, float], tuple[float, float]] = [(0.,0.),(0.,0.)]
) -> waveguide:
    '''
        optical.waveguide.rectangular
          initializes an optical waveguide with twisted rectangular shape.
    '''
    lx, ly = lengths;                                           # guide lengths of each coordinate
    phi_in, phi_out = angulations;                              # guide angulations on edges
    Lx = lx if callable(lx) else lambda z: lx;
    Ly = ly if callable(ly) else lambda z: ly;
    rect = lambda u, L: (np.abs(u) < 0.5 * L);
    omega = (phi_out - phi_in) / (zf - zi);
    cos_phi = lambda z: np.cos(phi_in + omega * (z - zi));
    sin_phi = lambda z: np.sin(phi_in + omega * (z - zi));
    return waveguide(
        delta_n = delta_n,
        function = lambda x, y, z: rect(
            cos_phi(z) * x - sin_phi(z) * y, Lx(z)
        ) * rect(
            sin_phi(z) * x + cos_phi(z) * y, Ly(z)
        ),
        center = curve(zi, zf, positions),
        zi = zi,
        zf = zf
    );

def cylindrical(
    delta_n: float,
    radius: float | np.ufunc,
    zi: float,
    zf: float,
    curve: tuple[np.ufunc, np.ufunc] = straight,
    positions: list[tuple[float, float], tuple[float, float]] = [(0.,0.),(0.,0.)],
    inhomogeneous: bool = False
) -> waveguide:
    '''
        optical.waveguide.cylindrical
          initializes an optical waveguide with cylindrical shape.
    '''
    R = radius if callable(radius) else lambda z: radius;
    if inhomogeneous:
        return waveguide(
            delta_n = delta_n,
            function = lambda r, _, z: np.exp(- (r / R(z)) ** 2.),
            center = curve(zi, zf, positions),
            zi = zi,
            zf = zf
        );
    else:
        rect = lambda u, R: (np.abs(u) < R);
        return waveguide(
            delta_n = delta_n,
            function = lambda r, _, z: rect(r, R(z)),
            center = curve(zi, zf, positions),
            zi = zi,
            zf = zf
        );

def anullar(
    delta_n: float,
    radii: tuple[float | np.ufunc, float | np.ufunc],
    zi: float,
    zf: float,
    curve: tuple[np.ufunc, np.ufunc] = straight,
    positions: list[tuple[float, float], tuple[float, float]] = [(0.,0.),(0.,0.)]
) -> waveguide:
    '''
        optical.waveguide.anullar
          initializes an optical waveguide with anullar shape.
    '''
    pass;