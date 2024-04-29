import numpy as np

class waveguide:
    def __init__(
        self,
        guide_geometry: np.ufunc,
        delta_n: float | np.ufunc,
        extension: tuple[float, float] = (-np.infty, +np.infty)
    ) -> None:
        '''
        ## `optical.medium.waveguides.waveguide`:
            creates an waveguide with `guide_geometry` and `extension` in z, furthermore
            `delta_n` as refractive index.
        
        ### syntax:
            `waveguide = optical.medium.waveguides.waveguide(geometry, extension = (z0, zf), delta_n)`
        
        ### parameters:
            `guide_geometry`: `numpy.ufunc`
                geometry of the waveguide.
            `delta_n`: `numpy.ufunc`
                refractive index variation through the waveguide.
            [optional] `extension`: `tuple[float,float]`
                float 2-tuple with `z0` and `zf` coordinates of waveguide.
        '''
        z0, zf = extension;
        if (z0 == -np.infty) and (zf == +np.infty):
            self.isInWG = lambda x,y,z: (guide_geometry(x,y,z));
        elif (z0 == -np.infty):
            self.isInWG = lambda x,y,z: (z <= zf) & (guide_geometry(x,y,z));
        elif (zf == +np.infty):
            self.isInWG = lambda x,y,z: (z0 <= z) & (guide_geometry(x,y,z));
        else:
            self.isInWG = lambda x,y,z: (z0 <= z <= zf) & (guide_geometry(x,y,z));
        self.delta_n = delta_n;
    def apply_refractive_index(
        self,
        region: tuple[np.ndarray, np.ndarray],
        z: np.ndarray | float
    ) -> np.ndarray:
        X, Y = region;
        if isinstance(self.delta_n, float):
            dn = self.delta_n * np.ones(X.shape);
        else:
            dn = self.delta_n(X, Y, z);
        return np.where(self.isInWG(X, Y, z), dn, 0.0);

# def visualize(waveguides: list[waveguide]) -> None:
#     pass

def straight_waveguide(
    initial_center: tuple[float, float],
    final_center: tuple[float, float],
    delta_n: float | np.ufunc,
    external_guide_waist: float | np.ufunc,
    internal_guide_waist: float | np.ufunc = 0.0,
    extension: tuple[float, float] = (-np.infty, +np.infty),
    center_refractive_index: bool = False
) -> waveguide:
    # compute spatial parameters
    x0, y0 = initial_center;
    xf, yf = final_center;
    z0, zf = extension;
    guideL = zf - z0;
    dir_vector = (
        (xf - x0) / guideL,
        (yf - y0) / guideL,
    );
    # guide geometry definition
    center_x = lambda z: x0 + dir_vector[0] * (z - z0);
    center_y = lambda z: y0 + dir_vector[1] * (z - z0);
    rho = lambda x, y, z: np.sqrt((x - center_x(z)) ** 2 + (y - center_y(z)) ** 2);
    if isinstance(internal_guide_waist, float) and isinstance(external_guide_waist, float):
        if internal_guide_waist == 0.0:
            guide_geometry = lambda x, y, z: rho(x, y, z) <= external_guide_waist;
        else:
            guide_geometry = lambda x, y, z: internal_guide_waist <= rho(x, y, z) <= external_guide_waist;
    elif isinstance(internal_guide_waist, float):
        if internal_guide_waist == 0.0:
            guide_geometry = lambda x, y, z: rho(x, y, z) <= external_guide_waist;
        else:
            guide_geometry = lambda x, y, z: internal_guide_waist <= rho(x, y, z) <= external_guide_waist;
    elif isinstance(external_guide_waist, float):
        guide_geometry = lambda x, y, z: internal_guide_waist(z) <= rho(x, y, z) <= external_guide_waist;
    else:
        guide_geometry = lambda x, y, z: internal_guide_waist(z) <= rho(x, y, z) <= external_guide_waist(z);
    # waveguide definition
    if isinstance(delta_n, float):
        return waveguide(
            guide_geometry = guide_geometry,
            delta_n = delta_n,
            extension = extension
        );
    else:
        if center_refractive_index:
            dn = lambda x, y, z: delta_n(x - center_x(z), y - center_y(z), z);
            return waveguide(
                guide_geometry = guide_geometry,
                delta_n = dn,
                extension = extension
            );
        else:
            return waveguide(
                guide_geometry = guide_geometry,
                delta_n = delta_n,
                extension = extension
            );

# def curved_waveguide(
#     initial_center: tuple[float, float],
#     final_center: tuple[float, float],
#     delta_n: float | np.ufunc,
#     external_guide_waist: float | np.ufunc,
#     internal_guide_waist: float | np.ufunc = 0.0,
#     extension: tuple[float, float] = (-np.infty, +np.infty),
#     center_refractive_index: bool = False
# ) -> waveguide: