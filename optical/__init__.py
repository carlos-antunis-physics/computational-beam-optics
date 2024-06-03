import numpy as np
from dataclasses import dataclass
from enum import Enum
from .medium import waveguide

'''
    coordinate systems in bidimensional space (transverse plane) transverse
    to axis of propagation (optical axis).
'''

@dataclass
class coordinate (Enum):
    cartesian: bool = True;
    polar: bool = False;

cart2pol = lambda x,y: (np.sqrt(x**2 + y**2), np.arctan2(y,x));
pol2cart = lambda r,theta: (r*np.cos(theta),r*np.sin(theta));

'''
    beam incidence through transverse plane.
'''

@dataclass
class wave_vector:
    x: np.float128;
    y: np.float128;
    def __init__(
        self,
        k: np.float128,
        angulation: tuple[np.float128,np.float128] = (0.0, 0.0)
    ) -> None:
        pi_by_180 = np.pi / 180.0;
        self.x = k * np.tan(angulation[0] * pi_by_180);
        self.y = k * np.tan(angulation[1] * pi_by_180);

'''
    optical medium generical construction.
'''

class medium:
    n0: np.float128;
    nonlinearity: np.ufunc;
    waveguides: list[waveguide];
    def __init__(
        self,
        n0: np.float128,
        nonlinearity: np.ufunc = lambda U: 0.0,
        waveguides: list[waveguide] = []
    ) -> None:
        '''
        ## `optical.medium.create`
            creates an optical medium with `n0` as base refractive index.
    
        ### syntax
            `medium = optical.medium(n0)`
        #### optional parameters
            `nonlinearity`: `numpy.ufunc`
                operator which encompass nonlinear response of the medium on
                light propagation.
            `waveguides`: `list[waveguides]`
                list of waveguides written in medium.
        '''
        self.refIndex = n0;
        self.__nonlinear_operator = nonlinearity;
        self.waveguides = waveguides;
    def apply_nonlinearity(self,U:np.ndarray) -> np.ndarray:
        '''
        ## `[optical.medium] medium.apply_nonlinearity`
            obtain the nonlinear behavior due to `U`.
    
        ### syntax
            `medium.apply_nonlinearity(U)`
        '''
        try:
            return self.__nonlinear_operator(U).astype(np.complex128);
        except AttributeError:
            return self.__nonlinear_operator(U);