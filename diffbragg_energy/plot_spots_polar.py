"""
Plot observed Bragg spot positions in polar coordinates derived from s1.

  radius  = crystallographic two-theta (angle between s1 and the incident beam s0)
  azimuth = angle of s1 projected onto the plane perpendicular to the beam

Spots are drawn as small open circles: red where dbi_lambda is NaN, green otherwise.

Usage (after sourcing a cctbx/DIALS setpaths script):
  libtbx.python plot_spots_polar.py file.expt file.refl [output.png]
"""
import sys
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dxtbx.model.experiment_list import ExperimentListFactory
from dials.array_family import flex
from scitbx.matrix import col


def lab_basis(beam):
    """Orthonormal (e1, e2, s0_hat) with e1, e2 spanning the plane normal to the beam.
    e1 follows the polarization normal (lab x for a standard setup)."""
    s0_hat = col(beam.get_unit_s0())
    pn = col(beam.get_polarization_normal())
    e1 = (pn - s0_hat * pn.dot(s0_hat)).normalize()
    e2 = s0_hat.cross(e1).normalize()
    return e1, e2, s0_hat


def main(expt_file, refl_file, out_png):
    experiments = ExperimentListFactory.from_json_file(expt_file, check_format=False)
    refls = flex.reflection_table.from_file(refl_file)

    two_theta_all, azimuth_all, isnan_all = [], [], []
    for i_expt, expt in enumerate(experiments):
        sel = refls["id"] == i_expt
        r = refls.select(sel)
        if len(r) == 0:
            continue
        e1, e2, s0_hat = lab_basis(expt.beam)

        s1 = r["s1"]
        s1_unit = s1 / s1.norms()   # direction only; wavelength-independent

        # two-theta: angle between scattered direction and incident beam
        cos2t = s1_unit.dot(s0_hat.elems)
        cos2t = flex.double([max(-1.0, min(1.0, c)) for c in cos2t])
        two_theta = flex.acos(cos2t) * 180.0 / math.pi

        # azimuth in the plane perpendicular to the beam
        x = s1_unit.dot(e1.elems)
        y = s1_unit.dot(e2.elems)
        azimuth = flex.atan2(y, x)

        lam = r["dbi_lambda"].as_numpy_array()
        two_theta_all.append(two_theta.as_numpy_array())
        azimuth_all.append(azimuth.as_numpy_array())
        isnan_all.append(np.isnan(lam))

    two_theta = np.concatenate(two_theta_all)
    azimuth = np.concatenate(azimuth_all)
    isnan = np.concatenate(isnan_all)

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    for mask, color, label in ((~isnan, "green", "dbi_lambda defined"),
                               (isnan, "red", "dbi_lambda = NaN")):
        ax.scatter(azimuth[mask], two_theta[mask], s=12, facecolors="none",
                   edgecolors=color, linewidths=0.8,
                   label="%s (%d)" % (label, mask.sum()))
    ax.set_rmin(0)
    ax.set_rlabel_position(135)
    ax.set_title("Bragg spots from s1: radius = 2θ (deg), angle = azimuth\n%s"
                 % refl_file, fontsize=9)
    ax.legend(loc="upper right", bbox_to_anchor=(1.12, 1.08), fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print("Plotted %d spots (%d green, %d red); max 2theta = %.2f deg -> %s"
          % (len(two_theta), (~isnan).sum(), isnan.sum(), two_theta.max(), out_png))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    out = sys.argv[3] if len(sys.argv) > 3 else "spots_polar.png"
    main(sys.argv[1], sys.argv[2], out)
