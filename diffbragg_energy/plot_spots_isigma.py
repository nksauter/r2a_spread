"""
Plot Bragg spot positions in polar coordinates derived from s1, filling each
circle with a color that encodes I/sigma(I) = intensity.sum.value / sqrt(intensity.sum.variance).

  radius  = crystallographic two-theta (angle between s1 and the incident beam s0)
  azimuth = angle of s1 projected onto the plane perpendicular to the beam

All spots are plotted (I/sigma does not depend on dbi_lambda) unless --only-dbi is given.
The color scale runs from min(I/sigma) to the --upper-percentile value; brighter spots
saturate at the top color.

Usage (after sourcing a cctbx/DIALS setpaths script):
  libtbx.python plot_spots_isigma.py file.expt file.refl [-o out.png] [--only-dbi]
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dxtbx.model.experiment_list import ExperimentListFactory
from dials.array_family import flex

from plot_spots_energy import spot_coordinates


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("expt")
    parser.add_argument("refl")
    parser.add_argument("-o", "--output", default="spots_isigma.png")
    parser.add_argument("--only-dbi", action="store_true",
                        help="plot only spots whose dbi_lambda is not NaN")
    parser.add_argument("--upper-percentile", type=float, default=99.0,
                        help="I/sigma percentile mapped to the top of the color scale")
    parser.add_argument("--cmap", default="viridis")
    args = parser.parse_args()

    experiments = ExperimentListFactory.from_json_file(args.expt, check_format=False)
    refls = flex.reflection_table.from_file(args.refl)
    # spot_coordinates walks experiments in id order; select the same rows here
    refls = flex.reflection_table.concat(
        [refls.select(refls["id"] == i) for i in range(len(experiments))])
    two_theta, azimuth, lam, _ = spot_coordinates(experiments, refls)

    inten = refls["intensity.sum.value"].as_numpy_array()
    var = refls["intensity.sum.variance"].as_numpy_array()
    with np.errstate(divide="ignore", invalid="ignore"):
        isig = inten / np.sqrt(var)

    keep = np.isfinite(isig) & (var > 0)
    if args.only_dbi:
        keep &= ~np.isnan(lam)

    vmin = isig[keep].min()
    vmax = np.percentile(isig[keep], args.upper_percentile)

    fig, ax = plt.subplots(figsize=(9.5, 8), subplot_kw={"projection": "polar"})
    order = np.argsort(isig[keep])  # strongest spots drawn on top
    sc = ax.scatter(azimuth[keep][order], two_theta[keep][order], c=isig[keep][order],
                    cmap=args.cmap, vmin=vmin, vmax=vmax, s=14,
                    edgecolors="black", linewidths=0.25)
    ax.set_rmin(0)
    ax.set_rlabel_position(135)
    ax.set_title("Bragg spots from s1: radius = 2θ (deg), angle = azimuth\n%s"
                 % args.refl, fontsize=9, pad=18)

    cbar = fig.colorbar(sc, ax=ax, orientation="vertical", pad=0.08, shrink=0.8,
                        extend="max")
    cbar.set_label("I / σ(I)   (intensity.sum.value / sqrt(intensity.sum.variance))")
    fig.savefig(args.output, dpi=150, bbox_inches="tight")

    v = isig[keep]
    print("Plotted %d spots (%s)" % (keep.sum(), "dbi_lambda only" if args.only_dbi else "all"))
    print("I/sigma: min %.2f  median %.2f  mean %.2f  max %.2f; color scale top (%.0fth pct) = %.2f"
          % (v.min(), np.median(v), v.mean(), v.max(), args.upper_percentile, vmax))
    print("Spots with I/sigma > 3: %d (%.1f%%)" % ((v > 3).sum(), 100.0 * (v > 3).mean()))
    print("-> %s" % args.output)


if __name__ == "__main__":
    main()
