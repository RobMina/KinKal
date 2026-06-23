#!/usr/bin/env python3
"""Scan the thick-material treatment across the Mu2e-relevant phase space.

Two passes:
  1. PREDICT-ONLY grid (Geant4-free, instant): material x thickness(X0) x momentum.
     Reports the new sub-stepped KinKal prediction vs the two-plane workaround vs the
     old single-step value -- i.e. WHERE the corrections matter across the space.
  2. GEANT4 validation subset: a handful of representative cells with full statistics,
     to confirm the predictions track Geant4 away from the original 1 GeV/c point.

The momentum sample points are drawn from the cosmic muon spectrum measured at the
tracker (prod_100k.root, truth |p| over TRK_SIDS): 10th/25th/50th/75th pct ~
0.6/1.3/3.2/7.9 GeV/c, deliberately weighted toward the soft end (0.3-2 GeV/c) where the
momentum-loss fraction -- and thus the thick-material correction -- is largest.

The "thickness" axis is expressed in radiation lengths and is read as PATH LENGTH, so it
also stands in for oblique incidence (path = t/cos(theta)) without a separate angle scan.

Run on a GPVM/build host from KinKal/Benchmarks after building ./build/ThickSlabG4.
"""
import subprocess, sys, os, math

BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build", "ThickSlabG4")

MATERIALS = ["CONCRETE_MARS", "ALUMINUM", "IRON"]
# momentum sample points (MeV/c), spectrum-representative with extra weight on the soft end
MOMENTA = [300.0, 600.0, 1000.0, 2000.0, 4000.0, 8000.0]
# thickness axis in radiation lengths (Mu2e concrete ~5-8 X0; DS steel ~tens; 20 = stress point)
X0_TARGETS = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]

# Geant4 validation subset: (material, x0_target, momentum). Momenta chosen above the muon
# range for that cell so enough primaries punch through for clean statistics.
G4_SUBSET = [
    ("CONCRETE_MARS", 1.0, 1000.0),
    ("CONCRETE_MARS", 5.0, 1000.0),
    ("CONCRETE_MARS", 5.0, 3000.0),
    ("CONCRETE_MARS", 10.0, 2000.0),
    ("IRON",          5.0, 1000.0),
    ("IRON",         10.0, 4000.0),
    ("ALUMINUM",      5.0, 1000.0),
]
G4_EVENTS = 2000

# Impact confirmation: run Geant4 for each material along the tracker->CRV-top extrapolation path
# at its ACTUAL Mu2eKinKal thickness (from KinKalGeomMaker / the run-2 geometry), so the per-material
# scattering deficit is weighted by the real radiation-length budget.  This shows how much the
# effective-Z bug actually biases the extrapolation, not just the per-material percentage.  CRV
# modules are omitted on purpose (extreme downstream end of the path; their scattering does not feed
# the prediction).  Thicknesses are the radial shell thickness / concrete depth at normal incidence.
# effz_scale = sqrt(S_true/S_eff) = the factor by which sigma_theta would rise if scatterAngleVar
# used the correct mass-weighted Z(Z+1)/A instead of the lumped effective Z (1.0 for a pure element).
#   (material, thickness_mm, momentum_MeVc, effz_scale, role)
MATERIAL_CONFIRM = [
    ("STAINLESSSTEEL",  44.0, 1000.0, 1.003, "DS cryostat shells (inner 19 + outer 25 mm)"),
    ("ALUMINUM",        25.4, 1000.0, 1.000, "DS thermal shields (2 x 12.7 mm, pure -> control)"),
    ("DSCOILMIX",       28.1, 1000.0, 1.091, "DS coils (effective radial thickness)"),
    ("CONCRETE_MARS",  447.0, 1000.0, 1.121, "ExtShield roof crossbar (~2/3 of tracks)"),
    ("CONCRETE_MARS",  904.0, 1000.0, 1.121, "ExtShield roof full T crossbar+stem (~1/3)"),
]
CONFIRM_EVENTS = 2000  # 2000 avoids the intermittent G4 malloc abort seen at higher counts; ~1.6% RMS stat error


def run(args):
    out = subprocess.run([BIN] + args, capture_output=True, text=True)
    if out.returncode != 0:
        sys.stderr.write(f"FAILED rc={out.returncode}: {' '.join(args)}\n{out.stderr[-400:]}\n")
        return None
    d = {}
    for line in out.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2:
            try:
                d[parts[0]] = float(parts[1])
            except ValueError:
                d[parts[0]] = parts[1]
    return d


def radlen_mm(material):
    """X0 (mm) for a material, from a single predict-only call (independent of t, p)."""
    d = run(["--predict-only", "--material", material, "--momentum", "1000", "--thickness", "100"])
    return d["kinkal_radlen_mm"]


def main():
    print(f"# binary: {BIN}")
    x0 = {m: radlen_mm(m) for m in MATERIALS}
    print("# radiation lengths (mm): " + ", ".join(f"{m}={x0[m]:.2f}" for m in MATERIALS))

    # ---- Pass 1: predict-only grid ----
    print("\n## PREDICT-ONLY GRID  (sig_theta and mean dE: new | 2plane | single-step ; vs nothing-G4)")
    hdr = ("material", "t/X0", "t(mm)", "p(MeV)", "kappa", "p_eff",
           "sth_new", "sth_2pl", "sth_1st", "dE_new", "dE_2pl", "dE_1st", "rms_new", "rms_2pl")
    print("\t".join(hdr))
    rows = []
    for mat in MATERIALS:
        for x0t in X0_TARGETS:
            t_mm = x0t * x0[mat]
            for p in MOMENTA:
                d = run(["--predict-only", "--material", mat,
                         "--momentum", f"{p:.0f}", "--thickness", f"{t_mm:.3f}"])
                if d is None:
                    continue
                row = (mat, f"{x0t:g}", f"{t_mm:.1f}", f"{p:.0f}",
                       f"{d['kinkal_vavilov_kappa']:.3f}", f"{d['kinkal_p_eff_MeVc']:.0f}",
                       f"{d['kinkal_sigma_theta']:.5f}", f"{d['twoplane_sigma_theta']:.5f}",
                       f"{d['kinkal_sigma_theta_singlestep']:.5f}",
                       f"{d['kinkal_mean_dE_MeV']:.2f}", f"{d['twoplane_mean_dE_MeV']:.2f}",
                       f"{d['kinkal_mean_dE_singlestep_MeV']:.2f}",
                       f"{d['kinkal_rms_dE_MeV']:.2f}", f"{d['twoplane_rms_dE_MeV']:.2f}")
                print("\t".join(row))
                rows.append(row)

    # ---- Pass 2: Geant4 validation subset ----
    print("\n## GEANT4 VALIDATION SUBSET  (events={}, no --max-step)".format(G4_EVENTS))
    g4hdr = ("material", "t/X0", "t(mm)", "p(MeV)", "exit_frac",
             "sth_new", "sth_2pl", "sth_g4", "dE_new", "dE_2pl", "dE_g4", "rms_new", "rms_g4")
    print("\t".join(g4hdr))
    for mat, x0t, p in G4_SUBSET:
        t_mm = x0t * x0[mat]
        d = run(["--material", mat, "--momentum", f"{p:.0f}", "--thickness", f"{t_mm:.3f}",
                 "--events", str(G4_EVENTS), "--seed", "24680"])
        if d is None:
            continue
        sth_g4 = 0.5 * (d["sigma_theta_x_g4"] + d["sigma_theta_y_g4"])
        sx_g4 = 0.5 * (d["sigma_x_g4_mm"] + d["sigma_y_g4_mm"])
        exitfrac = d["events_exited"] / d["events_requested"]
        print("\t".join((mat, f"{x0t:g}", f"{t_mm:.1f}", f"{p:.0f}", f"{exitfrac:.2f}",
                          f"{d['kinkal_sigma_theta']:.5f}", f"{d['twoplane_sigma_theta']:.5f}",
                          f"{sth_g4:.5f}",
                          f"{d['kinkal_mean_dE_MeV']:.2f}", f"{d['twoplane_mean_dE_MeV']:.2f}",
                          f"{d['mean_dE_g4_MeV']:.2f}",
                          f"{d['kinkal_rms_dE_MeV']:.2f}", f"{d['rms_dE_g4_MeV']:.2f}")))

    # ---- Pass 3: extrapolation-path impact (real Mu2eKinKal thicknesses) ----
    print("\n## EXTRAPOLATION-PATH SCATTERING IMPACT vs GEANT4  (events={}, real geometry thicknesses, p=1 GeV/c)".format(CONFIRM_EVENTS))
    print("# sigma_theta in mrad. sth_corr = KinKal with the effective-Z bug removed (x effz_scale).")
    print("# effZ_mr = sth_corr - sth_kk = mrad the effective-Z bug alone misses; resid_mr = g4 - corr (scatterfrac/thin-formula, not the bug).")
    print("\t".join(("material", "t(mm)", "t/X0", "exit", "sth_g4", "sth_kk", "sth_corr",
                     "effZ_mr", "resid_mr", "role")))
    s_g4=s_kk=s_corr=0.0  # quadrature over per-track path (steel+Al+coils+crossbar concrete)
    for mat, t_mm, p, effz, role in MATERIAL_CONFIRM:
        d = run(["--material", mat, "--momentum", f"{p:.0f}", "--thickness", f"{t_mm:.3f}",
                 "--events", str(CONFIRM_EVENTS), "--seed", "24680"])
        if d is None:
            continue
        sth_g4 = 0.5 * (d["sigma_theta_x_g4"] + d["sigma_theta_y_g4"])
        sth_kk = d["kinkal_sigma_theta"]
        sth_corr = sth_kk * effz
        toverx0 = t_mm / d["kinkal_radlen_mm"]
        exitfrac = d["events_exited"] / d["events_requested"]
        print("\t".join((mat, f"{t_mm:.1f}", f"{toverx0:.2f}", f"{exitfrac:.2f}",
                          f"{1e3*sth_g4:.3f}", f"{1e3*sth_kk:.3f}", f"{1e3*sth_corr:.3f}",
                          f"{1e3*(sth_corr-sth_kk):+.3f}", f"{1e3*(sth_g4-sth_corr):+.3f}", role)))
        if not (mat == "CONCRETE_MARS" and t_mm > 500):  # crossbar concrete only, 1x each layer
            s_g4 += sth_g4**2; s_kk += sth_kk**2; s_corr += sth_corr**2
    g4=1e3*math.sqrt(s_g4); kk=1e3*math.sqrt(s_kk); corr=1e3*math.sqrt(s_corr)
    print(f"# PATH TOTAL (steel+Al+coils+crossbar concrete, quadrature, 1 GeV/c):")
    print(f"#   G4 {g4:.2f} mrad | KinKal {kk:.2f} ({1-kk/g4:+.1%}) | effZ-corrected {corr:.2f} ({1-corr/g4:+.1%})")
    print(f"#   effective-Z bug alone: {corr-kk:+.2f} mrad ({(corr-kk)/g4:+.1%} of the path scattering)")


if __name__ == "__main__":
    main()
