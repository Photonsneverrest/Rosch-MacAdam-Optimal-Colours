# Project Title

Surface of colour solid is shown representing optimal colours. These are the colours an average human eye can perceive perceptually equally spaced.
It is used as reference to quantify the performance of a pigment. Two ratios are taken for a given hue: chroma effciency and lightness efficiency,
each as ratio of the from the spectrum calculated chroma and lightness divided by their at the same hue corresponding maximal values.


$$
\eta_C = \frac{C}{C(hue)}
$$

$$
\eta_{L*} = \frac{L*}{L*(hue)}
$$

L* is the CIELAB lightness value. Chroma is given through the euclidean norm in the a*-b* plane $\sqrt{a^2 + b^2}$ and can be understood as saturation.
Values in the denominator are maximum chroma values at same hue while the hue is the polar angle in the a*-b* plane $\theta = \arctan\!\left(\frac{b*}{a*}\right)$.
The maximum chroma values are shown in the gif below as bigger markers on the surface of the Rösch-MacAdam colour solid.
The chroma efficiency has its maximum at 100% since the colour solid seems to be convex. While the lightness efficiency has its optimum at 100% while when above approaching the white point, hence desaturating.


<p align="center">
  <img
    src="GIF/rosch_macadam_colour_solid_rotation_optimalcolours.gif"
    alt="MacAdam Color Solid Rotation with optimal colours"
    style="max-width: 100%; height: auto; width: 800px;">
  
  <p align="center">
  <img
    src="GIF/rosch_macadam_colour_solid_rotation_rgbebu_inf.gif"
    alt="MacAdam Color Solid and RGB-EBU"
    style="max-width: 100%; height: auto; width: 800px;">
---

## Features
- **Feature 1**: Computes and plots colour solid in CIELAB coordinate system.
- **Feature 2**: Shows rgb ebu colorspace within.
- **Feature 3**: Use from Spectrum_ColorProps import compute_color_properties to compute chroma and lightness performance of a spectrum respectively to its hue.

---

## Installation

## How to run
```bash
python main.py
