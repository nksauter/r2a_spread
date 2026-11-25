"""
Histogram of per-spot X-ray energies, E = h*c / dbi_lambda (spots with NaN dbi_lambda
are skipped). Vertical lines mark the mean spot energy and the beam energy from the .expt.

Usage (after sourcing a cctbx/DIALS setpaths script):
  libtbx.python plot_energy_histogram.py file.expt file.refl [-o out.png] [--bin-width 2] [--log]
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dxtbx.model.experiment_list import ExperimentListFactory
from dials.array_family import flex

from plot_spots_energy import wavelength_to_ev


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("expt")
    parser.add_argument("refl")
    parser.add_argument("-o", "--output", default="energy_histogram.png")
    parser.add_argument("--bin-width", type=float, default=2.0, help="bin width in eV")
    parser.add_argument("--log", action="store_true", help="logarithmic count axis")
    args = parser.parse_args()

    experiments = ExperimentListFactory.from_json_file(args.expt, check_format=False)
    refls = flex.reflection_table.from_file(args.refl)

    lam = refls["dbi_lambda"].as_numpy_array()
    good = ~np.isnan(lam)
    energy = wavelength_to_ev(lam[good])
    mean_energy = energy.mean()
    median_energy = np.median(energy)
    beam_energy = wavelength_to_ev([e.beam.get_wavelength() for e in experiments]).mean()

    lo = np.floor(energy.min() / args.bin_width) * args.bin_width
    hi = np.ceil(energy.max() / args.bin_width) * args.bin_width + args.bin_width
    bins = np.arange(lo, hi, args.bin_width)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(energy, bins=bins, color="0.55", edgecolor="black", linewidth=0.4,
            log=args.log)
    ax.axvline(mean_energy, color="black", linestyle="-", linewidth=1.5,
               label="mean spot energy %.1f eV" % mean_energy)
    ax.axvline(median_energy, color="black", linestyle=":", linewidth=1.5,
               label="median spot energy %.1f eV" % median_energy)
    ax.axvline(beam_energy, color="tab:red", linestyle="--", linewidth=1.5,
               label="beam energy (.expt) %.1f eV" % beam_energy)
    ax.set_xlabel("spot energy from dbi_lambda (eV)")
    ax.set_ylabel("number of spots per %.3g eV bin" % args.bin_width)
    ax.set_title("Spot energy histogram (%d spots, %d NaN skipped)\n%s"
                 % (good.sum(), (~good).sum(), args.refl), fontsize=9)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)

    p = np.percentile(energy, [1, 5, 25, 50, 75, 95, 99])
    print("Spots: %d (NaN skipped: %d)" % (good.sum(), (~good).sum()))
    print("Energy: min %.1f  max %.1f  mean %.1f  std %.1f eV; beam %.1f eV"
          % (energy.min(), energy.max(), mean_energy, energy.std(), beam_energy))
    print("Percentiles 1/5/25/50/75/95/99: " + "  ".join("%.1f" % x for x in p))
    print("-> %s" % args.output)


if __name__ == "__main__":
    main()
