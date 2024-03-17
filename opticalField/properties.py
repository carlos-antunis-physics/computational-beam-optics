import numpy as np

def intensity(U: np.ndarray) -> np.ndarray:
    return np.abs(U) ** 2.0;

def phase(U: np.ndarray) -> np.ndarray:
    return np.angle(U);