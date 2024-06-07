# COMPUTATIONAL BEAM OPTICS

Author: Carlos Antunis Bonfim da Silva Santos

![Linguagem Python](https://img.shields.io/badge/Linguagem%20Python-3572A5?style=plastic)
![numpy: 1.26.4](https://img.shields.io/badge/numpy-1.26.4-green?style=plastic)
![scipy: 8.2](https://img.shields.io/badge/scipy-8.2-green?style=plastic)
![matplotlib: 3.8.3](https://img.shields.io/badge/matplotlib-3.8.3-green?style=plastic)
![ipython: 8.22.2](https://img.shields.io/badge/ipython-8.22.2-green?style=plastic)

This repository is widely inspired on [FiniteDiferenceBPM](https://github.com/Windier/FiniteDifferenceBPM) repository, implemented by [José Carlos](https://github.com/Windier).

To obtain the library in this repo (ensuring that your computer satisfy the required dependencies), first, download the files in `main` branch, directly [here](https://github.com/carlos-antunis-physics/computational-beam-optics/archive/refs/heads/main.zip) or by `git`:

```bash
# obtain the computational-beam-optics repository
git clone 'https://github.com/carlos-antunis-physics/computational-beam-optics.git'
```

now, you have direct access to `computational-beam-optics/` directory with the library `optical/` and the `examples/` provided here.

## ABOUT THIS REPOSITORY

Understand how optical media (even more non-linear ones) affect the propagation of light beams are of enormous importance, besides an important source of novel applications. However, providing an analytical method which has the capability to encompass the effects (embedding also possible non-linear responses) in light propagation is quite difficult (or sometimes impossible).

The application of computational methods to estimate how the medium disturbs the light propagation is a widely used approach on the study of the optical phenomena. This repository aims to provide a python library with numerical routines to estimate the solutions of a Dirichlet Boundary Value Problem (BVP) defined by the inhomogeneous non-linear elliptical partial differential equations

$$
    \imath\partial_z\psi(\mathbf{r}, z) = \frac{1}{2\kappa}\nabla^2_\perp\psi(\mathbf{r}, z) + \Delta{n}(\mathbf{r},z)\psi(\mathbf{r},z) + \mathcal{N}(\psi(\mathbf{r},z))\psi(\mathbf{r},z)\text{,}
$$

namely here as non-linear and inhomogeneous Helmholtz paraxial equation, furthermore to the beam entry profile

$$
    \psi(\mathbf{r}, z = 0) = \psi_{z = 0}(\mathbf{r})\text{,}
$$

which are the boundary condition of our Dirichlet BVP.
