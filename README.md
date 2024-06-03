# COMPUTATIONAL BEAM OPTICS

> Author: Carlos Antunis Bonfim da Silva Santos

This repository is based on: [FiniteDiferenceBPM](https://github.com/Windier/FiniteDifferenceBPM) of [José Carlos](https://github.com/Windier)

## ABOUT THIS REPO

![Linguagem Python](https://img.shields.io/badge/Linguagem%20Python-3572A5?style=plastic)
![numpy: 1.26.4](https://img.shields.io/badge/numpy-1.26.4-green?style=plastic)
![scipy: 8.2](https://img.shields.io/badge/scipy-8.2-green?style=plastic)
![matplotlib: 3.8.3](https://img.shields.io/badge/matplotlib-3.8.3-green?style=plastic)
![ipython: 8.22.2](https://img.shields.io/badge/ipython-8.22.2-green?style=plastic)

Understand how an optical medium (even more the non-linear ones) affects the light beam propagation are of enormous importance, also being an important source of novel applications. However, provide an analytical method cappable to comprehend this (which embbed possible non-linear effects) is quite difficult (or sometimes impossible). 

The approach on the study of general optical phenomena are broadly the using of computational estimation methods. This repository aims to provide a python module with numerical routines to solve partial diferential equations which encompass the optical media effects (even if non-linear) on light propagation.

## USING THE LIBRARY

To obtain the library in this repo (ensure that your computer satisfy the dependencies), first, dowload the files in `main` branch:

```bash
wget "https://github.com/carlos-antunis-physics/computational-beam-optics/archive/refs/heads/main.zip"
```

subsequently, obtain the `optical` directory in the `main.zip` file:

```bash
unzip main.zip                      # unzip main branch zip file
# extract only computational-beam-optics-main/optical/
mv computational-beam-optics-main/optical/ ./optical/
rm -r computational-beam-optics-main/
```

the optical directory contains everything in the library - now just put in your project and use.
