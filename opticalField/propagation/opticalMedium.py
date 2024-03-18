import numpy as np

class waveGuide:
    def __init__(self, center, guideWaist, delta_n, z_i:float, z_f:float) -> None:
        """
            waveGuide:
                create waveguide centered in center(z) curve with guideWaist(z) of guide waist
                along z (between z_i and z_f) presenting delta_n(X, Y, z) of difference of
                base refractive index.

                syntax:
                    WG = waveGuide(center, guideWaist, delta_n, z_i, z_f);
                input arguments:
                    center: lambda z: tuple[float, float]
                        center of waveguide
                    guideWaist: lambda z: float
                        waist of waveguide
                    delta_n: lambda X,Y,x0,y0,z: float
                        base refractive index difference due to waveguide
                    z_i: float
                        waveguide initial z point
                    z_f: float
                        waveguide final z point
        """
        self.center = center;
        self.guideWaist = guideWaist;
        self.delta_n = delta_n;
        self.z_i = z_i;
        self.z_f = z_f;
    def applyRefractiveIndex(self, region:tuple[np.ndarray, np.ndarray], z:float) -> np.ndarray | float:
        if self.z_i <= z <= self.z_f:
            X, Y = region;
            xc_WG, yc_WG = self.center(z);
            R = self.guideWaist(z);
            isInWaveguide = lambda X, Y: (X - xc_WG) ** 2 + (Y - yc_WG) ** 2 <= R ** 2;
            return np.where(
                isInWaveguide(X, Y),
                self.delta_n(       # if is in waveguide return diference in refractive index
                    X, Y, xc_WG, yc_WG, z
                ),
                0.0                 # else there are not diference in refractive index due to waveguide
            );
        else:
            return 0.0;

def straightWaveguide(
    center_i: tuple[float, float],
    center_f: tuple[float, float],
    guideWaist,
    delta_n,
    z_i: float,
    z_f: float
) -> waveGuide:
    # compute procedure parameters
    x_i, y_i = center_i;
    x_f, y_f = center_f;
    # compute straight line director vector
    dirVec_x, dirVec_y = (x_f - x_i), (y_f - y_i);
    Dz = z_f - z_i;
    dirVec_x /= Dz;
    dirVec_y /= Dz;
    # construct straight line curve
    straightLine = lambda z: (x_i + dirVec_x * (z - z_i), y_i + dirVec_y * (z - z_i));
    return waveGuide(
        center = straightLine,
        guideWaist = guideWaist,
        delta_n = delta_n,
        z_i = z_i,
        z_f = z_f
    );