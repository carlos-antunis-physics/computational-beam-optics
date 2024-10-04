'''
    useful python packages
'''

# external python imports
import numpy as np

# interal optical python module imports
from optical import waveguide
from optical.waveguides.utils import straight

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
    positions: list[tuple[float, float], tuple[float, float]] = [(0.,0.),(0.,0.)],
    variation: float | np.ufunc = 1.
) -> waveguide:
    '''
        optical.waveguide.rectangular
          initializes an optical waveguide with rectangular shape.
    '''
    lx, ly = lengths;                                           # guide lengths of each coordinate
    Lx = lx if callable(lx) else lambda z: lx;
    Ly = ly if callable(ly) else lambda z: ly;
    _var = variation if callable(variation) else lambda z: variation;
    rect = lambda u, L: (np.abs(u) < 0.5 * L);
    cos_phi, sin_phi = np.cos(angulation), np.sin(angulation);
    return waveguide(
        delta_n = delta_n,
        function = lambda x, y, z: _var(z) * rect(
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
    positions: list[tuple[float, float], tuple[float, float]] = [(0.,0.),(0.,0.)],
    variation: float | np.ufunc = 1.
) -> waveguide:
    '''
        optical.waveguide.rectangular
          initializes an optical waveguide with twisted rectangular shape.
    '''
    lx, ly = lengths;                                           # guide lengths of each coordinate
    phi_in, phi_out = angulations;                              # guide angulations on edges
    Lx = lx if callable(lx) else lambda z: lx;
    Ly = ly if callable(ly) else lambda z: ly;
    _var = variation if callable(variation) else lambda z: variation;
    rect = lambda u, L: (np.abs(u) < 0.5 * L);
    omega = (phi_out - phi_in) / (zf - zi);
    cos_phi = lambda z: np.cos(phi_in + omega * (z - zi));
    sin_phi = lambda z: np.sin(phi_in + omega * (z - zi));
    return waveguide(
        delta_n = delta_n,
        function = lambda x, y, z: _var(z) * rect(
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
    variation: float | np.ufunc = 1.,
) -> waveguide:
    '''
        optical.waveguide.cylindrical
          initializes an optical waveguide with cylindrical shape.
    '''
    R = radius if callable(radius) else lambda z: radius;
    _var = variation if callable(variation) else lambda z: variation;
    rect = lambda u, R: (np.abs(u) < R);
    return waveguide(
        delta_n = delta_n,
        function = lambda r, _, z: _var(z) * rect(r, R(z)),
        center = curve(zi, zf, positions),
        zi = zi,
        zf = zf
    );

def inhomogeneous_cylindrical(
    delta_n: float,
    radius: float | np.ufunc,
    zi: float,
    zf: float,
    curve: tuple[np.ufunc, np.ufunc] = straight,
    positions: list[tuple[float, float], tuple[float, float]] = [(0.,0.),(0.,0.)],
    variation: float | np.ufunc = 1.,
) -> waveguide:
    '''
        optical.waveguide.cylindrical
          initializes an inhomogeneous optical waveguide with cylindrical shape.
    '''
    R = radius if callable(radius) else lambda z: radius;
    _var = variation if callable(variation) else lambda z: variation;
    return waveguide(
        delta_n = delta_n,
        function = lambda r, _, z:  _var(z) * np.exp(- (r / R(z)) ** 2.),
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
    positions: list[tuple[float, float], tuple[float, float]] = [(0.,0.),(0.,0.)],
    variation: float | np.ufunc = 1.
) -> waveguide:
    '''
        optical.waveguide.anullar
          initializes an optical waveguide with anullar shape.
    '''
    r_0, r_f = radii;
    R_0 = r_0 if callable(r_0) else lambda z: r_0;
    R_f = r_f if callable(r_f) else lambda z: r_f;
    _var = variation if callable(variation) else lambda z: variation;
    rect = lambda u, r, R: (np.abs(u) > r) * (np.abs(u) < R);
    return waveguide(
        delta_n = delta_n,
        function = lambda r, _, z: _var(z) * rect(r, R_0(z), R_f(z)),
        center = curve(zi, zf, positions),
        zi = zi,
        zf = zf
    );