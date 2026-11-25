### Simple scripts to see the distribution of energies listed in Derek's diffBragg analysis. ###

New per-spot wavelengths are in the "dbi_lambda" column of the refl file.

```
libtbx.python plot_spots_polar.py *.expt *.refl
Plotted 2226 spots (2182 green, 44 red); max 2theta = 65.08 deg -> spots_polar.png
```
Shows a polar plot with failed energy determination in red, success in green
```
libtbx.python plot_spots_energy.py *.expt *.refl
hc = 12398.419843 eV*A
I/sigma cutoff 1: 606 of 2226 spots pass
Plotted 593 spots with dbi_lambda; 13 passing NaN spots omitted
Spot energy: min 7073.9  max 7180.7  mean 7097.2  std 13.3 eV
Percentiles 5/25/50/75/95: 7084.5 / 7091.1 / 7095.4 / 7098.5 / 7114.7 eV
Color scale 7079.7 .. 7114.7 eV (mean +/- 17.5); pegged red: 9, pegged blue: 30
Beam energy: 7090.5 eV (lambda = 1.74860 A)
-> spots_energy.png
```
This shows that the mean 1-sigma spot energy is about 7 eV greater than the beam energy. 
The peak energy (giving rise to 1 sigma or greater spots, is higher than the mean beam.
```
libtbx.python plot_spots_isigma.py *.expt *.refl
Plotted 2226 spots (all)
I/sigma: min -3.44  median 0.23  mean 0.66  max 58.23; color scale top (99th pct) = 10.76
Spots with I/sigma > 3: 132 (5.9%)
-> spots_isigma.png
```
Shows there are a few 2 sigma spots all the way to the edge
```
libtbx.python plot_energy_histogram.py *.expt *.refl
Spots: 2182 (NaN skipped: 44)
Energy: min 7072.3  max 7180.8  mean 7097.3  std 14.1 eV; beam 7090.5 eV
Percentiles 1/5/25/50/75/95/99: 7076.5  7082.0  7091.2  7095.6  7099.1  7119.2  7176.2
-> energy_histogram.png
```
