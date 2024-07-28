# The `optical` module

The structure of `optical` module is the following

> **`optical`**
>
> base methods, objects and utilitaries for setup elements of light propagation on optical media estimation.
>
> > **`Beam`**
> >
> > methods for evaluate canonical transverse profiles of beam optics.
> >
> > > **`Propagation`**
> > >
> > > numerical methods from estimate the propagation of light beams.
> > > 
> > > > **`boundary`**
> > > >
> > > > base objects for apply boundary conditions on methods of light propagation estimation.
> > > 
> 
> > **`Waveguide`**
> >
> > base methods to facilitate optical waveguide setup for estimation of light propagation.
>
> > **`Utils`**
> >
> > utils methods for computational research of optics.
>

## methods of `optical`

> `optical.beam`
> > evaluate the transverse profile of an optical beam along a region on the transverse plane.

> `optical.wave_number`
> > evaluate the wave number for a specified wave length.

> `optical.rayleigh_range`
> > evaluate the Rayleigh range.

> `optical.beam_waist`
> > evaluate the beam waist $W(z) = w_0\sqrt{1 + (z/z_0)^2}$.

> `optical.radius_of_curvature`
> > evaluate the radius of curvature $R(z) = z\sqrt{1 + (z_0/z)^2}$.

> `optical.gouy_phase`
> > evaluate the beam waist $\zeta(z) =$ arctg $(z/z_0)$.

> `optical.oblique_phasor`
> > evaluate the wave vector for the given angulations.

> `class optical.waveguide`
> > initializes an optical waveguide with the most generic properties along z axis.

> `class optical.medium`
> > initializes an optical medium with waveguides and non-linear responses as space.

### methods of `optical.Beam`

> `optical.Beam.gaussian`
> > evaluate a gaussian beam transverse profile on a region of transverse plane.

> `optical.Beam.hermite_gauss`
> > evaluate a hermite-gauss beam transverse profile on a region of transverse plane.

> `optical.Beam.laguerre_gauss`
> > evaluate a laguerre-gauss beam transverse profile on a region of transverse plane.

> `optical.Beam.bessel`
> > evaluate a bessel beam transverse profile on a region of transverse plane.

#### methods of `optical.Beam.Propagation`

> `optical.Beam.Propagation.split_step`
> > estimate the field propagated along z points on medium by split-step beam propagation method.

> **[work in progress]** `optical.Beam.Propagation.crank_nicolson`
> > estimate the field propagated along z points on medium using finite-diferences.

##### methods of `optical.Beam.Propagation.boundary`

> `optical.Beam.Propagation.boundary.absorbing`
> > initializes an absorbing layer to act as boundary condition on propagation methods.

> **[work in progress]**  `optical.Beam.Propagation.boundary.transparent`
> > initializes a transparent boundary to act as boundary condition on propagation methods.

> **[work in progress]** `optical.Beam.Propagation.boundary.perfectly_matched_layer`
> initializes an perfectly matched layer to act as boundary condition on propagation methods.

### methods of `optical.Waveguide`

> `optical.Waveguide.straight`
> > construct the curve as a straight line to center a waveguide.

> `optical.Waveguide.curved`
> > construct the curve as a curved line to center a waveguide.

> `optical.Waveguide.rectangular`
> > initializes a simple optical waveguide with rectangular shape.

> `optical.Waveguide.cylindrical`
> > initializes a simple optical waveguide with cylindrical shape.

> `optical.Waveguide.gaussian_cylindrical`
> > initializes an optical waveguide with cylindrical shape whose index fade as a gaussian radially.

### methods of `optical.Utils`

> `optical.Utils.thomas`
> > solves the linear system of equations in which coefficients is a tridiagonal matrix (used at `optical.Beam.Propagation.crank_nicolson` method).
