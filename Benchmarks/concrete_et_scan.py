#!/usr/bin/env python3
"""Post-fix residual map for CONCRETE_MARS over the momentum x thickness (E/t) grid.

For every cell that physically transmits the muon, run Geant4 (reference) and the post-fix KinKal
thick-scatter prediction, and report the residual (KinKal - G4)/G4 for the three observables:
  sigma_theta (scattering width, now effective-Z corrected), mean dE, RMS dE.
Range-limited cells (muon stops) are skipped.  Run from KinKal/Benchmarks after building.
"""
import subprocess, os

BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build", "ThickSlabG4")
MOMENTA = [300.0, 600.0, 1000.0, 2000.0, 4000.0, 8000.0]   # MeV/c
X0S     = [0.5, 1.0, 2.0, 5.0, 10.0]                        # thickness in radiation lengths
EVENTS  = 2000
# crude muon CSDA range in concrete (mm) to skip range-out cells (don't waste G4 / risk the crash)
RANGE_MM = {300: 230, 600: 900, 1000: 2070, 2000: 4440, 4000: 9000, 8000: 18000}


def run(args):
    out = subprocess.run([BIN] + args, capture_output=True, text=True)
    if out.returncode != 0:
        return None
    d = {}
    for line in out.stdout.splitlines():
        p = line.split()
        if len(p) == 2:
            try: d[p[0]] = float(p[1])
            except ValueError: d[p[0]] = p[1]
    return d


def main():
    x0 = run(["--predict-only", "--material", "CONCRETE_MARS",
              "--momentum", "1000", "--thickness", "100"])["kinkal_radlen_mm"]
    print(f"# CONCRETE_MARS post-fix residual map  (X0={x0:.2f} mm, events={EVENTS})")
    print(f"# residual = (KinKal - G4)/G4 ; sth = sigma_theta (eff-Z corrected)")
    print("\t".join(("p(MeV)", "t/X0", "t(mm)", "exit", "sth_kk", "sth_g4",
                     "sth_resid", "dE_resid", "rms_resid")))
    for p in MOMENTA:
        for x in X0S:
            t_mm = x * x0
            if RANGE_MM[int(p)] < 1.05 * t_mm:
                print("\t".join((f"{p:.0f}", f"{x:g}", f"{t_mm:.0f}", "-", "-", "-",
                                 "range-out", "range-out", "range-out")))
                continue
            d = run(["--material", "CONCRETE_MARS", "--momentum", f"{p:.0f}",
                     "--thickness", f"{t_mm:.3f}", "--events", str(EVENTS), "--seed", "24680"])
            if d is None:
                print("\t".join((f"{p:.0f}", f"{x:g}", f"{t_mm:.0f}", "CRASH", "-", "-", "-", "-", "-")))
                continue
            sth_g4 = 0.5 * (d["sigma_theta_x_g4"] + d["sigma_theta_y_g4"])
            sth_kk = d["kinkal_sigma_theta"]
            exitf = d["events_exited"] / d["events_requested"]
            sth_r = (sth_kk - sth_g4) / sth_g4
            de_r = (d["kinkal_mean_dE_MeV"] - d["mean_dE_g4_MeV"]) / d["mean_dE_g4_MeV"]
            rms_r = (d["kinkal_rms_dE_MeV"] - d["rms_dE_g4_MeV"]) / d["rms_dE_g4_MeV"]
            print("\t".join((f"{p:.0f}", f"{x:g}", f"{t_mm:.0f}", f"{exitf:.2f}",
                             f"{sth_kk:.5f}", f"{sth_g4:.5f}",
                             f"{sth_r:+.1%}", f"{de_r:+.1%}", f"{rms_r:+.1%}")))


if __name__ == "__main__":
    main()
