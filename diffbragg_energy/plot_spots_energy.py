"""
Plot Bragg spot positions in polar coordinates derived from s1, filling each
circle with a color that encodes the per-spot X-ray energy from dbi_lambda.

  radius  = crystallographic two-theta (angle between s1 and the incident beam s0)
  azimuth = angle of s1 projected onto the plane perpendicular to the beam
  fill    = photon energy E = h*c / dbi_lambda, red (lowest eV) -> white -> blue (highest eV)
          The scale is linear and symmetric about the mean spot energy (white, center
          of the color bar). Its half-width is the larger of (mean - 5th percentile) and
          (95th percentile - mean), so both the 5% and 95% points are on the scale and one
          sits at an end. Spots beyond the scale are still drawn, pegged at full red or blue.
          Hashmarks on the color bar show the 5/25/50/75/95% points of the spot energies.

Only spots with I/sigma(I) = intensity.sum.value / sqrt(intensity.sum.variance) above
--sigma (default 1.0) are plotted. The colorbar on the right marks the mean energy of those
spots and the beam energy from the .expt.
Spots whose dbi_lambda is NaN are omitted unless --show-nan is given (the I/sigma cutoff
applies to them too).

Usage (after sourcing a cctbx/DIALS setpaths script):
  libtbx.python plot_spots_energy.py file.expt file.refl [-o out.png] [--sigma 1.0] [--show-nan]
"""
import argparse
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np

from dxtbx.model.experiment_list import ExperimentListFactory
from dials.array_family import flex
from scitbx.matrix import col

# E [eV] = h c / (e * lambda); CODATA 2018 exact SI values
PLANCK_H = 6.62607015e-34        # J s
SPEED_OF_LIGHT = 299792458.0     # m / s
ELEMENTARY_CHARGE = 1.602176634e-19  # C (J per eV)
HC_EV_ANGSTROM = PLANCK_H * SPEED_OF_LIGHT / ELEMENTARY_CHARGE * 1.0e10  # ~12398.42 eV*A


def wavelength_to_ev(lam_angstrom):
    return HC_EV_ANGSTROM / np.asarray(lam_angstrom, dtype=float)


def lab_basis(beam):
    """Orthonormal (e1, e2, s0_hat) with e1, e2 spanning the plane normal to the beam.
    e1 follows the polarization normal (lab x for a standard setup)."""
    s0_hat = col(beam.get_unit_s0())
    pn = col(beam.get_polarization_normal())
    e1 = (pn - s0_hat * pn.dot(s0_hat)).normalize()
    e2 = s0_hat.cross(e1).normalize()
    return e1, e2, s0_hat


def spot_coordinates(experiments, refls):
    """Return two_theta (deg), azimuth (rad), dbi_lambda (A) and the beam wavelengths."""
    tt_all, az_all, lam_all, beam_lams = [], [], [], []
    for i_expt, expt in enumerate(experiments):
        r = refls.select(refls["id"] == i_expt)
        if len(r) == 0:
            continue
        beam_lams.append(expt.beam.get_wavelength())
        e1, e2, s0_hat = lab_basis(expt.beam)

        s1_unit = r["s1"] / r["s1"].norms()
        cos2t = s1_unit.dot(s0_hat.elems).as_numpy_array().clip(-1.0, 1.0)
        tt_all.append(np.degrees(np.arccos(cos2t)))
        x = s1_unit.dot(e1.elems).as_numpy_array()
        y = s1_unit.dot(e2.elems).as_numpy_array()
        az_all.append(np.arctan2(y, x))
        lam_all.append(r["dbi_lambda"].as_numpy_array())
    return (np.concatenate(tt_all), np.concatenate(az_all),
            np.concatenate(lam_all), np.array(beam_lams))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("expt")
    parser.add_argument("refl")
    parser.add_argument("-o", "--output", default="spots_energy.png")
    parser.add_argument("--sigma", type=float, default=1.0,
                        help="plot only spots with I/sigma(I) above this cutoff (default 1.0)")
    parser.add_argument("--show-nan", action="store_true",
                        help="also plot spots whose dbi_lambda is NaN (red, unfilled)")
    parser.add_argument("--cmap", default="bwr_r",
                        help="matplotlib colormap; default bwr_r = red(low E)-white-blue(high E)")
    args = parser.parse_args()

    experiments = ExperimentListFactory.from_json_file(args.expt, check_format=False)
    refls = flex.reflection_table.from_file(args.refl)
    # spot_coordinates walks experiments in id order; put the table rows in the same order
    refls = flex.reflection_table.concat(
        [refls.select(refls["id"] == i) for i in range(len(experiments))])
    two_theta, azimuth, lam, beam_lams = spot_coordinates(experiments, refls)

    inten = refls["intensity.sum.value"].as_numpy_array()
    var = refls["intensity.sum.variance"].as_numpy_array()
    with np.errstate(divide="ignore", invalid="ignore"):
        isig = inten / np.sqrt(var)
    strong = np.isfinite(isig) & (var > 0) & (isig > args.sigma)

    good = ~np.isnan(lam) & strong
    if not good.any():
        raise SystemExit("No spots with dbi_lambda and I/sigma > %g" % args.sigma)
    energy = wavelength_to_ev(lam[good])
    mean_energy = energy.mean()
    beam_energies = wavelength_to_ev(beam_lams)
    beam_energy = beam_energies.mean()

    # Symmetric linear scale about the mean, wide enough to show both the 5% and 95% points
    pct_levels = (5, 25, 50, 75, 95)
    pct_values = np.percentile(energy, pct_levels)
    half_width = max(mean_energy - pct_values[0], pct_values[-1] - mean_energy)
    vmin, vmax = mean_energy - half_width, mean_energy + half_width
    norm = Normalize(vmin=vmin, vmax=vmax, clip=False)  # out-of-range -> cmap under/over
    n_below, n_above = (energy < vmin).sum(), (energy > vmax).sum()
    extend = {(False, False): "neither", (True, False): "min",
              (False, True): "max", (True, True): "both"}[(n_below > 0, n_above > 0)]

    fig, ax = plt.subplots(figsize=(9.5, 8), subplot_kw={"projection": "polar"})

    # Draw in order of distance from the mean so strongly colored spots sit on top.
    order = np.argsort(np.abs(energy - mean_energy))
    sc = ax.scatter(azimuth[good][order], two_theta[good][order], c=energy[order],
                    cmap=args.cmap, norm=norm,
                    s=14, edgecolors="black", linewidths=0.25,
                    label="spots with dbi_lambda (%d)" % good.sum())

    if args.show_nan:
        bad = np.isnan(lam) & strong
        ax.scatter(azimuth[bad], two_theta[bad], s=14, facecolors="none",
                   edgecolors="red", linewidths=0.8,
                   label="dbi_lambda = NaN (%d)" % bad.sum())
        ax.legend(loc="upper left", bbox_to_anchor=(-0.12, 1.08), fontsize=8)

    ax.set_rmin(0)
    ax.set_rlabel_position(135)
    ax.set_title("Bragg spots from s1: radius = 2θ (deg), angle = azimuth;  "
                 "I/σ(I) > %g\n%s" % (args.sigma, args.refl), fontsize=9, pad=18)

    cbar = fig.colorbar(sc, ax=ax, orientation="vertical", pad=0.08, shrink=0.8,
                        extend=extend)
    cbar.set_label("spot energy from dbi_lambda (eV)")

    # Percentile hashmarks: short ticks on the left edge of the bar, labelled outside it
    for level, value in zip(pct_levels, pct_values):
        cbar.ax.plot([-0.35, 0.45], [value, value], color="black", linewidth=1.2,
                     transform=cbar.ax.get_yaxis_transform(), clip_on=False)
        cbar.ax.annotate("%d%%" % level, xy=(-0.35, value), xycoords=("axes fraction", "data"),
                         xytext=(-2, 0), textcoords="offset points",
                         ha="right", va="center", fontsize=8)

    # Mean and beam energy as lines across the bar, explained in a legend under the plot
    cbar.ax.axhline(mean_energy, color="black", linestyle="-", linewidth=1.5)
    if vmin <= beam_energy <= vmax:
        cbar.ax.axhline(beam_energy, color="black", linestyle="--", linewidth=1.5)
    beam_note = "" if vmin <= beam_energy <= vmax else " (off scale)"
    handles = [
        Line2D([], [], color="black", linestyle="-", linewidth=1.5,
               label="mean spot energy (I/\u03c3>%g): %.1f eV" % (args.sigma, mean_energy)),
        Line2D([], [], color="black", linestyle="--", linewidth=1.5,
               label="beam energy (.expt): %.1f eV%s" % (beam_energy, beam_note)),
        Line2D([], [], color="black", linestyle="-", linewidth=1.2,
               label="hashmarks: %s%% of spot energies (%s eV)"
                     % ("/".join(str(l) for l in pct_levels),
                        " / ".join("%.1f" % v for v in pct_values))),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=1, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.06))

    fig.savefig(args.output, dpi=150, bbox_inches="tight")

    print("hc = %.6f eV*A" % HC_EV_ANGSTROM)
    print("I/sigma cutoff %g: %d of %d spots pass" % (args.sigma, strong.sum(), len(isig)))
    print("Plotted %d spots with dbi_lambda; %d passing NaN spots %s"
          % (good.sum(), (np.isnan(lam) & strong).sum(),
             "shown" if args.show_nan else "omitted"))
    print("Spot energy: min %.1f  max %.1f  mean %.1f  std %.1f eV"
          % (energy.min(), energy.max(), mean_energy, energy.std()))
    print("Percentiles %s: %s eV" % ("/".join(str(l) for l in pct_levels),
                                     " / ".join("%.1f" % v for v in pct_values)))
    print("Color scale %.1f .. %.1f eV (mean +/- %.1f); pegged red: %d, pegged blue: %d"
          % (vmin, vmax, half_width, n_below, n_above))
    if len(beam_energies) > 1:
        print("Beam energy (mean of %d experiments): %.1f eV" % (len(beam_energies), beam_energy))
    else:
        print("Beam energy: %.1f eV (lambda = %.5f A)" % (beam_energy, beam_lams[0]))
    print("-> %s" % args.output)


if __name__ == "__main__":
    main()
