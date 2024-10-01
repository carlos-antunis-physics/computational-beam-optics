# Computational beam optics

![Python](https://img.shields.io/badge/Python-3572A5?style=plastic)
![Fortran](https://img.shields.io/badge/Fortran-4d41b1?style=plastic)
![numpy: 1.26.4](https://img.shields.io/badge/numpy-1.26.4-green?style=plastic)
![scipy: 8.2](https://img.shields.io/badge/scipy-8.2-green?style=plastic)
![matplotlib: 3.8.3](https://img.shields.io/badge/matplotlib-3.8.3-green?style=plastic)
![ipython: 8.22.2](https://img.shields.io/badge/ipython-8.22.2-green?style=plastic)
![scikit-image: 0.24.0](https://img.shields.io/badge/scikit--image-0.24.0-green?style=plastic)

> Author: Carlos Antunis Bonfim da Silva Santos
>
> This repository is widely inspired on [FiniteDiferenceBPM](https://github.com/Windier/FiniteDifferenceBPM) repository, implemented by [José Carlos](https://github.com/Windier).

## Computational estimation of light propagation

The understanding of how light propagation is affected by an optical media (even more those with non-linear responses) is of enormous importance, besides an important source of novel applications. The applying of numerical methods to estimate how the disturbs on the light propagation due an optical medium occurs is widely used to approach on the study of nonlinear and waveguide optics, since providing an analytical method which encompass both the linear and non-linear effects is quite difficult (or, in most of the cases, impossible).

> ### About the repository
>
> > This repository aims to provide a python library with a collection of computational algorithms used for research on modern optics. Besides some utils algorithms, the most of the algorithms arranged here consists of methods to estimate the solutions of a Dirichlet Boundary Value Problem (BVP) for inhomogeneous non-linear elliptical partial differential equations such as
> >
> > $$
> >     \jmath\partial_z\psi(\textbf{r},z) = \frac{1}{2\kappa}\nabla^2_\perp\psi(\textbf{r},z) + \Delta{n}(\textbf{r},z)\psi(\textbf{r},z) + \mathrm{N}(\psi(\textbf{r},z))\psi(\textbf{r},z)\text{,}
> > $$
> > 
> > defined here as non-linear and inhomogeneous Helmholtz paraxial equation.

> ### Using the repository
>
> > To obtain the library `optical` provided by this repository (firstly ensure that your computer satisfy the required dependencies) download the files on the `main` branch, directly [here](https://github.com/carlos-antunis-physics/computational-beam-optics/archive/refs/heads/main.zip) or using `git`:
> >
> > ```bash
> > git clone 'https://github.com/carlos-antunis-physics/computational-beam-optics.git'
> > ```
> >
> > than obtain the`optical/` directory (it's all that you need to perform simulations) at `computational-beam-optics/`.
> >
> > ```bash
> > mv ./computational-beam-optics/optical/ ./optical/
> > rm -r ./computational-beam-optics/
> > ```
> >
> > > To compile `optical/Utils/linear_algebra.f95` to a python module, use:
> > >
> > > ```bash
> > > f2py -m linear_algebra linear_algebra.f95 -h linear_algebra.pyf --overwrite-signature
> > > f2py -c linear_algebra.pyf linear_algebra.f95
> > > ```
> >
<!-- > > for usage references, see the [`examples`](./examples/) (will be updated soon) provided here or the [`documentation`](./documentation.md). -->
