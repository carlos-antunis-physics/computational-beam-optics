# Computational beam optics

Authors: [Carlos Antunis Bonfim da S. Santos](https://github.com/carlos-antunis-physics) & [Carlos Eduardo da S. Santana](https://github.com/caduAa)

![Python](https://img.shields.io/badge/Python-3572A5?style=plastic)
![Fortran](https://img.shields.io/badge/Fortran-4d41b1?style=plastic)
![numpy: 1.26.4](https://img.shields.io/badge/numpy-1.26.4-3572A5?style=plastic)
![scipy: 8.2](https://img.shields.io/badge/scipy-8.2-3572A5?style=plastic)
![matplotlib: 3.8.3](https://img.shields.io/badge/matplotlib-3.8.3-3572A5?style=plastic)
![ipython: 8.22.2](https://img.shields.io/badge/ipython-8.22.2-3572A5?style=plastic)
![scikit-image: 0.24.0](https://img.shields.io/badge/scikit--image-0.24.0-3572A5?style=plastic)

## Computational optics python module

Understanding how to properly control light, besides a scientific research area of enormous impact, is an important source of novel technological applications. Provide a general analytical method which also encompass non-linear effects of optical media is impossible, thus numerical methods are widely applied both to estimate how light propagation is disturbed by optical responses (even the non-linear ones) of optical devices and design themselves.

### About this python module

> The python module developed in this repo is widely inspired on [FiniteDiferenceBPM](https://github.com/Windier/FiniteDifferenceBPM) repository, implemented by [José Carlos do A. Rocha](https://github.com/Windier).

Our main goal is to provide a python module with a wide range of computational methods used in our academic researchs at the **Optics and Nanoscopy Group** ([GON](https://if.ufal.br/grupopesquisa/gon/index_en.html)). Besides some utils algorithms of phase optimization, the most of the algorithms arrenged here consists of methods to estimate the solutions of a Dirichlet Boundary Value Problem (BVP) for inhomogeneous non-linear elliptical partial differential equations such as

$$
    \jmath\partial_z\psi(\textbf{r},z) = \frac{1}{2\kappa}\nabla^2_\perp\psi(\textbf{r},z) + \Delta{n}(\textbf{r},z)\psi(\textbf{r},z) + \mathrm{N}(\psi(\textbf{r},z))\psi(\textbf{r},z)\text{,}
$$

defined here as non-linear and inhomogeneous Helmholtz paraxial equation.

### Installing the python module on your personal computer

To obtain the python module `optical` provided by this repository, ensuring that your computer satisfy the required dependencies (listed in the [top of this `README`](#computational-beam-optics)), you will need only the `optical/` directory  as the python module on your python application at the `main` branch.

<!-- ```bash
f2py -m linear_algebra linear_algebra.f95 -h linear_algebra.pyf --overwrite-signature
f2py -c linear_algebra.pyf linear_algebra.f95
``` -->

#### Direct download

Download the files on `main` branch directly [here](https://github.com/carlos-antunis-physics/computational-beam-optics/archive/refs/heads/main.zip).

#### Download using `git`

Download the files on `main` branch using `git`:

```bash
git clone 'https://github.com/carlos-antunis-physics/computational-beam-optics.git'
```

#### Installing the python module in a `Google Colab notebook`

To obtain the python module `optical` provided by this repository on a `Google Colab notebook` add a cell and use the commands:

```python
from google.colab import output

#  change directory to google colab root directory
%cd -q /content/

# clone git repository
!apt-get install git
%rm -r /content/computational-beam-optics/ /content/optical/
!git clone 'https://github.com/carlos-antunis-physics/computational-beam-optics'

# obtain python module directory
%mv ./computational-beam-optics/optical/ ./optical/
%rm -r ./computational-beam-optics/                 # remove other files

# enable widget visualization
!pip install ipympl
output.enable_custom_widget_manager()
```

<!-- ### Documentation and examples

For usage references, see the [`examples`](./examples/1-elementary/simple.ipynb) provided in this repository or check the [`documentation`](./documentation/main.md). -->