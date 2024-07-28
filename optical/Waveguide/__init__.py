'''
    importing of useful python packages
'''

import numpy as np

'''
    importing of optical module utils
'''

# import waveguide constructor
from optical import waveguide

'''
    simple methods for centering waveguides
'''

def straight(
    zi: float,
    zf: float,
    positions: list[tuple[float, float], tuple[float, float]]
) -> np.ufunc:
    '''
        optical.Waveguide.straight
            evaluate a straight line to center a waveguide.
    '''
    r_i, r_f = positions;                       # initial and final positions
    r_i = np.array(r_i); r_f = np.array(r_f);
    dz, delta_r = zf - zi, r_f - r_i;           # variations along path
    return (
        lambda z: r_i[0] + delta_r[0] * (z - zi) / dz,
        lambda z: r_i[1] + delta_r[1] * (z - zi) / dz
    );

def curved(
    zi: float,
    zf: float,
    positions: list[tuple[float, float], tuple[float, float]]
) -> np.ufunc:
    '''
        optical.Waveguide.curved
            evaluate a straight line to center a waveguide.
    '''
    r_i, r_f = positions;                       # initial and final positions
    r_i = np.array(r_i); r_f = np.array(r_f);
    dz, delta_r = zf - zi, r_f - r_i;           # variations along path
    d = float(np.sqrt(sum(delta_r ** 2.)));
    if d == 0.: 
        return straight(zi,zf, positions);
    else:
        phi = np.arctan2(delta_r[1], delta_r[0]);
        R = (dz ** 2. + d ** 2.) / (4. * d);
        ax = r_i[0] + R;
        bx = d - R + r_i[0];
        sin_phi, cos_phi = np.sin(phi), np.cos(phi);
        half_dz = dz / 2.;
        sg_d = np.sign(d);
        def x(z: float | np.ndarray):
            if isinstance(z, float):
                if (z <= zi + half_dz):
                    return (-sg_d * np.sqrt(R ** 2. - (z - zi) ** 2.) + ax);
                else:
                    return (+sg_d * np.sqrt(R ** 2. - (z - zf) ** 2.) + bx);
            else:
                return np.where(
                    z <= zi + half_dz,
                    -sg_d * np.sqrt(R ** 2. - (z - zi) ** 2.) + ax,
                    +sg_d * np.sqrt(R ** 2. - (z - zf) ** 2.) + bx
                );
        return (
            lambda z: (cos_phi * (x(z) - r_i[0]) + sin_phi * r_i[1]) + r_i[0],
            lambda z: (sin_phi * (x(z) - r_i[0]) - cos_phi * r_i[1])
        );

'''
    simple optical waveguides simetries
'''

def rectangular(
    delta_n: float,
    lengths: tuple[float | np.ufunc, float | np.ufunc] | float | np.ufunc,
    zi: float,
    zf: float,
    curve: tuple[np.ufunc, np.ufunc] = straight,
    positions: list[tuple[float, float], tuple[float, float]] = [(0.,0.),(0.,0.)]
) -> waveguide:
    '''
        optical.Waveguide.rectangular
          initializes an optical waveguide with rectangular shape.
    '''
    lx, ly = lengths;                           # guide lengths of each coordinate
    Lx = lx if callable(lx) else lambda z: lx;
    Ly = ly if callable(ly) else lambda z: ly;
    rect = lambda u, L: ((np.abs(u) / L) < 0.5);
    return waveguide(
        delta_n = delta_n,
        function = lambda x, y, z: rect(x, Lx(z)) * rect(y, Ly(z)),
        center = curve(zi,zf, positions),
        zi = zi,
        zf = zf
    );

def cylindrical(
    delta_n: float,
    radius: float | np.ufunc,
    zi: float,
    zf: float,
    curve: tuple[np.ufunc, np.ufunc] = straight,
    positions: list[tuple[float, float], tuple[float, float]] = [(0.,0.),(0.,0.)]
) -> waveguide:
    '''
        optical.Waveguide.cylindrical
          initializes an optical waveguide with cylindrical shape.
    '''
    R = radius if callable(radius) else lambda z: radius;
    rect = lambda u, L: ((np.abs(u) / L) < 0.5);
    return waveguide(
        delta_n = delta_n,
        function = lambda r, phi, z: rect(r, R(z)),
        center = curve(zi,zf, positions),
        zi = zi,
        zf = zf
    );

def gaussian_cylindrical(
    delta_n: float,
    radius: float | np.ufunc,
    zi: float,
    zf: float,
    curve: tuple[np.ufunc, np.ufunc] = straight,
    positions: list[tuple[float, float], tuple[float, float]] = [(0.,0.),(0.,0.)]
) -> waveguide:
    '''
        optical.Waveguide.gaussian_cylindrical
          initializes an optical waveguide with cylindrical shape with gaussian radial
          fade.
    '''
    R = radius if callable(radius) else lambda z: radius;
    return waveguide(
        delta_n = delta_n,
        function = lambda r, phi, z: np.exp(- (r / R(z)) ** 2.),
        center = curve(zi,zf, positions),
        zi = zi,
        zf = zf
    );