"""
helix_wave_timing.py
====================

Predicted 1-D plate-impact wave timing for HELIX/SPADE spall traces.

Given the *measured* peak free-surface velocity of a trace plus the flyer and
target material Hugoniots and thicknesses, this module reconstructs the impact
kinematics and the characteristic wave-timing structure that a spall-free
(elastic/plastic, no-spall) record should show:

  1. Back-calculate the impact velocity V by impedance matching the measured
     free-surface state (Option 2 of the HELIX design discussion). No laser
     energy calibration is required -- everything comes from the trace itself.
  2. Shock speeds U_s in flyer and target from the linear Hugoniot U_s = C0 + S*u_p.
  3. Free-surface plateau (pulse) duration Δt_FS -- flyer round-trip plus the
     target-transit correction. Uses flyer thickness, and target thickness when
     available.
  4. Second-recompression timing -- the target round-trip that reloads the free
     surface after the release/pullback. Needs TARGET thickness.

Accuracy: this is a FIRST-ORDER, constant-wave-speed estimate. Two standing
approximations dominate its error and it should be read as an estimate, not an
exact timing:
  * Free-surface approximation u_p ~= u_fs/2 (release adiabat ~ Hugoniot).
  * Release / reverberation transit at a constant C_L. The C_L supplied in the
    HELIX config are ordinary ambient longitudinal speeds, used here as a proxy
    for the shocked-state Lagrangian release speed. At finite shock strength a
    release-adiabat / method-of-characteristics (MOC) treatment is needed for
    quantitative timings, and the interface-reflection SIGN (z_ratio) is only a
    heuristic (see impedance_ratio).
The impact-velocity reconstruction (steps 1-2) is exact under u_p ~= u_fs/2.

All physics functions take plain material-property dicts so the module stays
decoupled from HELIX config loading:

    prop = {'density': kg/m3, 'C0': m/s, 'C_L': m/s, 'S': dimensionless}

`C0` is the bulk (Hugoniot intercept) sound speed used for U_s = C0 + S*u_p.
`C_L` is the longitudinal sound speed used as the release/reverberation transit
speed (see accuracy note above). `S` is the Hugoniot slope.

Thicknesses are passed in micrometres (as stored in the parameter files) and
converted internally.  All returned times are in nanoseconds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Required keys for a material-property dict used by this module.
REQUIRED_KEYS = ("density", "C0", "C_L", "S")


class WaveTimingError(ValueError):
    """Raised when inputs are insufficient/unphysical for a timing prediction."""


def _validate_props(prop: dict, role: str) -> dict:
    """Return a clean {density, C0, C_L, S} dict or raise with a clear message."""
    if not isinstance(prop, dict):
        raise WaveTimingError(f"{role} properties must be a dict, got {type(prop)!r}")
    out = {}
    for key in REQUIRED_KEYS:
        val = prop.get(key)
        if val is None:
            raise WaveTimingError(f"{role} properties missing required key {key!r}")
        try:
            out[key] = float(val)
        except (TypeError, ValueError):
            raise WaveTimingError(f"{role} property {key!r}={val!r} is not numeric")
        if out[key] <= 0:
            raise WaveTimingError(f"{role} property {key!r} must be positive, got {out[key]}")
    return out


# ---------------------------------------------------------------------------
# Impedance matching / impact kinematics
# ---------------------------------------------------------------------------

def hugoniot_pressure(prop: dict, up: float) -> float:
    """Shock pressure P = rho0 * U_s * u_p with U_s = C0 + S*u_p (Pa)."""
    us = prop["C0"] + prop["S"] * up
    return prop["density"] * us * up


def impact_velocity_from_ufs(u_fs_peak: float, flyer: dict, target: dict) -> dict:
    """Back-calculate impact velocity from the measured peak free-surface velocity.

    Uses the free-surface approximation u_p(target) ~= u_fs / 2, then solves the
    flyer Hugoniot (centred at the moving flyer) for the impact velocity V that
    reproduces the same interface pressure:

        rho0_t * (C0_t + S_t*u_p) * u_p  ==  rho0_f * (C0_f + S_f*x) * x ,
        with x = V - u_p  (particle-velocity jump in the flyer).

    This is a quadratic in x; the single positive root is physical.

    Parameters
    ----------
    u_fs_peak : float
        Measured peak (plateau) free-surface velocity, m/s.
    flyer, target : dict
        Material property dicts with keys density, C0, C_L, S.

    Returns
    -------
    dict with keys:
        V                 impact velocity (m/s)
        up_interface      interface particle velocity == u_p(target) (m/s)
        up_flyer_jump     x = V - up_interface (m/s)
        P_interface       interface pressure (Pa)
        Us_target, Us_flyer   shock speeds (m/s)
    """
    flyer = _validate_props(flyer, "flyer")
    target = _validate_props(target, "target")
    if u_fs_peak is None or not np.isfinite(u_fs_peak) or u_fs_peak <= 0:
        raise WaveTimingError(f"peak free-surface velocity must be positive, got {u_fs_peak!r}")

    up_i = 0.5 * float(u_fs_peak)              # free-surface doubling approximation
    p_target = hugoniot_pressure(target, up_i)  # interface pressure from target side

    # Solve rho0_f*S_f*x^2 + rho0_f*C0_f*x - p_target = 0  for x > 0.
    a = flyer["density"] * flyer["S"]
    b = flyer["density"] * flyer["C0"]
    c = -p_target
    disc = b * b - 4 * a * c
    if disc < 0:
        raise WaveTimingError("no real flyer velocity reproduces the measured free-surface state")
    x = (-b + math.sqrt(disc)) / (2 * a)       # positive root
    if x <= 0:
        raise WaveTimingError("back-calculated flyer velocity jump is non-positive")

    V = up_i + x
    return {
        "V": V,
        "up_interface": up_i,
        "up_flyer_jump": x,
        "P_interface": p_target,
        "Us_target": target["C0"] + target["S"] * up_i,
        "Us_flyer": flyer["C0"] + flyer["S"] * x,
    }


def impedance_ratio(flyer: dict, target: dict, up_flyer: float, up_target: float) -> float:
    """Incremental (acoustic) impedance ratio Z_flyer / Z_target, used only to
    pick the SIGN of the interface reflection for the reverberation.

    A small release/recompression reverberating about a shocked state reflects
    according to the incremental acoustic impedance Z = rho_H * c = rho0 * C_L,
    NOT the shock (Rayleigh) impedance rho0*Us. Because this module stores C_L
    as the (Lagrangian) release-transit speed and uses h/C_L for transit, the
    consistent impedance is rho0 * C_L -- mixing a Lagrangian C_L with the
    shocked density rho_H would double-count the compression factor. The stored
    C_L is an ambient longitudinal speed standing in for the shocked-state
    Lagrangian release-head speed, so this ratio is an approximate SIGN guide,
    not a quantitative reflection coefficient (a release EOS is needed for that).

    The Lagrangian release speed is the Hugoniot-tangent proxy C0 + 2*S*up
    (state-aware, derivable from C0,S,up), and the consistent impedance is then
    Z = rho0 * C_L (derivation Sec. 7/8).

    <1  -> flyer is the softer (lower-impedance) side: a target-side release
           reflects off the interface as a *compression* (R_sigma < 0).
    >1  -> flyer is the stiffer side.
    """
    cL_f = flyer["C0"] + 2.0 * flyer["S"] * up_flyer     # Hugoniot-tangent Lagrangian proxy
    cL_t = target["C0"] + 2.0 * target["S"] * up_target
    z_f = flyer["density"] * cL_f
    z_t = target["density"] * cL_t
    return z_f / z_t


# ---------------------------------------------------------------------------
# Wave timing
# ---------------------------------------------------------------------------

@dataclass
class WaveTiming:
    """Predicted wave-timing structure, times in ns, referenced to shock arrival
    at the free surface (t=0), matching SPADE's time-shifted convention.

    Follows the isolated-path constant-speed construction of the HELIX timing
    derivation, but the DRAWN plateau uses the robust flyer round-trip
        plateau_duration_ns = t_flyer = h_f/Us_f + h_f/C_L_f
    rather than min(t_FR, P_t). The target-transit term in t_FR (Eq. 20) is a
    fragile difference amplified by h_t; with only an ambient C_L_t it is
    unreliable (Eq. 21-22) and can spuriously collapse the plateau for a thick
    target, whereas t_flyer is C_L_t-independent and matches observed plateau
    widths. t_FR (Eq. 20) and P_t (Eq. 24) are retained as ambient-C_L
    diagnostics, with the two mutually-exclusive release paths (Sec. 5)
    classified by `regime`:
      'no_target' : h_t unknown -> only the flyer round-trip is known.
      'catch_up'  : t_FR <= 0   -> (unreliable branch) ambient-C_L t_FR implies
                    shock-release catch-up; plateau still drawn at t_flyer.
      'case_A'    : 0 < t_FR < P_t -> release-fan interaction; the n*P_t
                    recompression series is NOT valid (suppressed); P_t is a
                    diagnostic marker only.
      'case_B'    : t_FR > P_t  -> initial free-surface release returns first;
                    for a low-impedance flyer (z_ratio<1) an isolated
                    recompression series t_rc,n = n*P_t is drawn (diagnostic).
    """
    u_fs_peak: float
    V: float
    up_interface: float
    Us_flyer: float
    Us_target: float
    cL_flyer_tangent: float             # Hugoniot-tangent Lagrangian release proxy C0f + 2 Sf x  (Eq. 41)
    cL_target_tangent: float            # C0t + 2 St u
    h_flyer_um: float
    h_target_um: Optional[float]
    plateau_duration_ns: float          # t_first = min(t_FR, P_t): first free-surface departure (Eq. 44)
    t_flyer_ns: float                   # flyer round-trip: h_f/Us_f + h_f/cL_f
    t_FR_ns: float                      # free-flight flyer-release arrival t_FR (Eq. 43)
    target_round_trip_ns: Optional[float]      # P_t = 2*h_t/cL_t (Eq. 42; None if h_t unknown)
    shock_arrival_ns: Optional[float]   # T_s = h_t/Us_t: flight time contact->target free surface (not the shifted-trace origin)
    spall_band_start_ns: Optional[float]   # candidate pullback onset at FS, rel. shock arrival (Sec. 10.2)
    spall_band_end_ns: Optional[float]     # candidate pullback window end (assumed fan width)
    regime: str                         # 'no_target' | 'catch_up' | 'case_A' | 'case_B'
    recompression_period_ns: Optional[float]   # P_t when a recompression series is drawn, else None
    recompression_times_ns: list = field(default_factory=list)  # t_rc,n = n*P_t (case_B, z<1 only)
    z_ratio: Optional[float] = None
    recompression_expected: bool = True
    notes: list = field(default_factory=list)


def compute_wave_timing(
    u_fs_peak: float,
    flyer: dict,
    target: dict,
    h_flyer_um: float,
    h_target_um: Optional[float] = None,
    n_recompressions: int = 2,
    fan_width_fraction: float = 0.15,
) -> WaveTiming:
    """First-hand constant-speed timing markers for a plate-impact spall trace.

    Follows the HELIX derivation Sec. 9-10 (first-hand overlay). Release and
    reverberation legs use the state-aware Hugoniot-tangent Lagrangian proxy
        cL_i = C0_i + 2 S_i u_p,i                                    (Eq. 41)
    (= rho0^-1 dsigma_H/du_p), NOT an ambient tabulated C_L. This is a bulk-
    family surrogate for the shocked-state release-head speed; the true release
    speed needs a release adiabat / MOC (Sec. 4). Here u_p is taken from the
    u_fs/2 back-calc (NOT first-hand: it uses the measured peak, flagged below).

    Markers (all relative to target shock arrival at the free surface):
      t_flyer = h_f/Us_f + h_f/cL_f                     flyer round-trip
      t_FR    = t_flyer + h_t*(1/cL_t - 1/Us_t)          Eq. 43
      P_t     = 2 h_t / cL_t                             Eq. 42
      t_first = min(t_FR, P_t)                           Eq. 44 (first FS departure)
    `regime` records the branch (Eq. 44 rule): 'catch_up' (t_FR<=0),
    'case_A' (0<t_FR<P_t: interacting releases, no n*P_t series), 'case_B'
    (t_FR>P_t: isolated recompression series n*P_t drawn if z_ratio<1).

    Candidate spall band (Sec. 10.2): the opposed leading release characteristics
    (flyer-release-in at the interface vs target free-surface release) intersect
    inside the target and the resulting pullback propagates to the free surface
    (Eqs. 48-52). A fan_width_fraction sets a trailing (slower) characteristic
    speed (1-f)*cL_t, giving the band end; the band is [start, end] rel. shock
    arrival. This is a coarse release-overlap window, not a spall-time prediction.

    Parameters
    ----------
    u_fs_peak : float          measured peak free-surface velocity (m/s)
    flyer, target : dict       material property dicts (density, C0, C_L, S)
    h_flyer_um : float         flyer thickness (micrometres)
    h_target_um : float | None target thickness (micrometres); None -> flyer-only
    n_recompressions : int     how many target round-trips to place (case_B only)
    fan_width_fraction : float assumed leading/trailing release-fan spread (0..1)

    Returns
    -------
    WaveTiming
    """
    flyer = _validate_props(flyer, "flyer")
    target = _validate_props(target, "target")
    if h_flyer_um is None or not np.isfinite(h_flyer_um) or h_flyer_um <= 0:
        raise WaveTimingError(f"flyer thickness must be positive, got {h_flyer_um!r}")

    kin = impact_velocity_from_ufs(u_fs_peak, flyer, target)
    us_f = kin["Us_flyer"]
    us_t = kin["Us_target"]
    up_i = kin["up_interface"]
    x_f = kin["up_flyer_jump"]

    # Physical-compression check (Us - up > 0) for both shocked states; a linear
    # Hugoniot with S<1 can violate this at high drive (derivation Sec. 3).
    if us_f - x_f <= 0 or us_t - up_i <= 0:
        raise WaveTimingError("unphysical shock state (Us - up <= 0); check C0/S at this drive")

    # Hugoniot-tangent Lagrangian release proxies (Eq. 41).
    cL_f = flyer["C0"] + 2.0 * flyer["S"] * x_f
    cL_t = target["C0"] + 2.0 * target["S"] * up_i

    h_f = h_flyer_um * 1e-6
    to_ns = 1e9

    # Flyer round-trip: shock in at Us_f, release back at cL_f.
    t_flyer_s = h_f / us_f + h_f / cL_f
    notes = [
        "impact velocity back-calculated from the measured peak (u_p=u_fs/2) -- "
        "NOT a first-hand/independent prediction (derivation Sec. 9)."
    ]

    h_t = None
    t_FR_s = t_flyer_s
    first_departure_s = t_flyer_s
    period_ns = None
    P_t_ns = None
    T_s_ns = None
    spall_start_ns = None
    spall_end_ns = None
    recompression_times = []
    z_ratio = None
    recompression_expected = False
    regime = "no_target"

    if h_target_um is not None and np.isfinite(h_target_um) and h_target_um > 0:
        h_t = h_target_um * 1e-6
        z_ratio = impedance_ratio(flyer, target, x_f, up_i)
        recompression_expected = z_ratio < 1.0

        # t_FR (Eq. 43), P_t (Eq. 42), t_first (Eq. 44).
        t_FR_s = t_flyer_s + h_t * (1.0 / cL_t - 1.0 / us_t)
        P_t_s = 2.0 * h_t / cL_t
        P_t_ns = P_t_s * to_ns
        first_departure_s = min(t_FR_s, P_t_s) if t_FR_s > 0 else 0.0

        # Regime branch (Eq. 44 rule).
        if t_FR_s <= 0:
            regime = "catch_up"
            notes.append(
                f"catch-up: t_FR={t_FR_s * to_ns:+.0f} ns <= 0 (release overtakes the "
                f"shock inside the target); no sustained plateau is physically valid."
            )
        elif t_FR_s < P_t_s:
            regime = "case_A"
            notes.append(
                f"Case A (t_FR={t_FR_s * to_ns:.0f} ns < P_t={P_t_ns:.0f} ns): interacting "
                f"releases -> no n*P_t recompression series; P_t drawn as a diagnostic marker."
            )
        else:
            regime = "case_B"
            if recompression_expected:
                period_ns = P_t_ns
                recompression_times = [n * P_t_ns for n in range(1, n_recompressions + 1)]
            else:
                notes.append(
                    f"Case B but flyer is the stiffer side "
                    f"(Z_flyer/Z_target={z_ratio:.2f}>=1): target-side return is a "
                    f"re-shock/unloading, not a low-impedance recompression."
                )

        # --- Candidate spall band (Sec. 10.2) ---------------------------------
        # Global times from first contact (T=0): target shock reaches the free
        # surface at T_s; flyer rear-surface release reaches the impact interface
        # at T_i. Opposed leading releases then cross inside the target.
        T_s = h_t / us_t
        T_i = h_f / us_f + h_f / cL_f
        T_s_ns = T_s * to_ns

        def _pullback_after_arrival(c):
            # Intersection of flyer-release-in (from X=0, launched T_i) and
            # target free-surface release (from X=h_t, launched T_s), both at
            # target release speed c; then propagate the pullback to the FS.
            if c <= 0:
                return None
            t_cap = h_t / (2.0 * c) + 0.5 * (T_i + T_s)
            x_cap = c * (t_cap - T_i)
            if not (0.0 < x_cap < h_t):
                return None
            t_pb = t_cap + (h_t - x_cap) / c
            return (t_pb - T_s) * to_ns   # referenced to shock arrival at FS

        lead = _pullback_after_arrival(cL_t)                              # leading (fast)
        trail = _pullback_after_arrival((1.0 - fan_width_fraction) * cL_t)  # trailing (slow)
        band = [b for b in (lead, trail) if b is not None and b >= 0]
        if band:
            spall_start_ns = min(band)
            spall_end_ns = max(band)
    else:
        notes.append("target thickness unavailable: only the flyer round-trip is known; recompression/spall omitted.")

    return WaveTiming(
        u_fs_peak=float(u_fs_peak),
        V=kin["V"],
        up_interface=up_i,
        Us_flyer=us_f,
        Us_target=us_t,
        cL_flyer_tangent=cL_f,
        cL_target_tangent=cL_t,
        h_flyer_um=float(h_flyer_um),
        h_target_um=(float(h_target_um) if h_t is not None else None),
        plateau_duration_ns=first_departure_s * to_ns,
        t_flyer_ns=t_flyer_s * to_ns,
        t_FR_ns=t_FR_s * to_ns,
        target_round_trip_ns=P_t_ns,
        shock_arrival_ns=T_s_ns,
        spall_band_start_ns=spall_start_ns,
        spall_band_end_ns=spall_end_ns,
        regime=regime,
        recompression_period_ns=period_ns,
        recompression_times_ns=recompression_times,
        z_ratio=z_ratio,
        recompression_expected=recompression_expected,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Idealized profile for plot overlay
# ---------------------------------------------------------------------------

def build_idealized_profile(
    timing: WaveTiming,
    t_anchor_ns: float = 0.0,
    rise_ns: float = 1.0,
    release_ns: float = 3.0,
    pullback_fraction: float = 0.85,
    recompression_fraction: float = 0.35,
    n_points: int = 600,
    t_end_ns: Optional[float] = None,
) -> tuple:
    """Build a schematic predicted free-surface velocity(t) for overlay.

    The overlay marks the FIRST-ORDER predicted timing (plateau width Δt_FS,
    release onset, recompression instants; see compute_wave_timing caveats) and
    is only schematic in amplitude on the release/recompression legs, which
    depend on details beyond this 1-D constant-speed model. It is anchored to
    the detected shock arrival (t_anchor_ns) so it lines up with the SPADE trace.
    After the release the released baseline is held flat, so passing t_end_ns
    (e.g. the trace's last time sample) extends the curve across the whole plot
    instead of stopping a few ns after the release.

    Parameters
    ----------
    timing : WaveTiming
    t_anchor_ns : float          shock-arrival time in the trace's shifted axis
    rise_ns : float              cosmetic rise time to the peak
    release_ns : float           cosmetic release (unloading) time constant
    pullback_fraction : float    dip depth as a fraction of peak (schematic)
    recompression_fraction : float  recompression bump height above the dip (schematic)
    n_points : int               samples in the returned arrays
    t_end_ns : float | None      extend the curve out to at least this time (ns)

    Returns
    -------
    (t_ns, v_ms) : np.ndarray, np.ndarray
    """
    peak = timing.u_fs_peak
    tau = timing.plateau_duration_ns
    t0 = t_anchor_ns

    t_rise_end = t0 + rise_ns
    t_plateau_end = t_rise_end + tau
    dip_v = peak * (1.0 - pullback_fraction)

    # Time span: cover through the last recompression (or the release tail), then
    # extend to t_end_ns if given so the released baseline spans the whole trace.
    if timing.recompression_times_ns:
        t_end = t_rise_end + timing.recompression_times_ns[-1] + 4 * release_ns
    else:
        t_end = t_plateau_end + 6 * release_ns
    if t_end_ns is not None and np.isfinite(t_end_ns):
        t_end = max(t_end, float(t_end_ns))
    t = np.linspace(t0 - 2 * rise_ns, t_end, n_points)
    v = np.zeros_like(t)

    # Rise (raised-cosine for a smooth foot).
    rise = (t >= t0) & (t < t_rise_end)
    v[rise] = peak * 0.5 * (1 - np.cos(np.pi * (t[rise] - t0) / max(rise_ns, 1e-9)))
    # Plateau.
    plat = (t >= t_rise_end) & (t < t_plateau_end)
    v[plat] = peak
    # Release: exponential decay from peak toward dip_v.
    rel = t >= t_plateau_end
    v[rel] = dip_v + (peak - dip_v) * np.exp(-(t[rel] - t_plateau_end) / max(release_ns, 1e-9))

    # Recompression bumps (Gaussians at n*P_t). recompression_times_ns are
    # referenced to shock arrival at the free surface; in the trace, that
    # reference is t_rise_end (the start of the held peak). Non-empty only in
    # case_B with a low-impedance flyer.
    for i, dt in enumerate(timing.recompression_times_ns):
        t_rc = t_rise_end + dt
        amp = peak * recompression_fraction * (0.7 ** i)  # decaying staircase
        width = max(release_ns, 1e-9)
        v += amp * np.exp(-0.5 * ((t - t_rc) / width) ** 2)

    return t, v


if __name__ == "__main__":
    # Self-test: Al (100 um) flyer on Brass target, using config Hugoniot values.
    al = {"density": 2700.0, "C0": 5240.0, "C_L": 6000.0, "S": 1.34}
    brass = {"density": 8520.0, "C0": 3800.0, "C_L": 4500.0, "S": 1.43}

    u_fs = 400.0  # m/s peak free-surface velocity (example)
    kin = impact_velocity_from_ufs(u_fs, al, brass)
    print(f"u_fs={u_fs} m/s  ->  V={kin['V']:.1f} m/s, "
          f"up_iface={kin['up_interface']:.1f}, "
          f"Us_flyer={kin['Us_flyer']:.0f}, Us_target={kin['Us_target']:.0f}")

    def _report(label, tim):
        _r = lambda v: v if v is None else round(v, 1)
        print(f"\n[{label}] regime={tim.regime}")
        print(f"  cL_f(tangent)={tim.cL_flyer_tangent:.0f}  cL_t(tangent)={tim.cL_target_tangent:.0f}")
        print(f"  t_flyer={tim.t_flyer_ns:.1f} ns  t_FR={tim.t_FR_ns:.1f} ns  P_t={_r(tim.target_round_trip_ns)} ns")
        print(f"  t_first (plateau end)={tim.plateau_duration_ns:.1f} ns  Z_f/Z_t={_r(tim.z_ratio)}")
        print(f"  spall band=[{_r(tim.spall_band_start_ns)}, {_r(tim.spall_band_end_ns)}] ns  "
              f"recompression={['%.1f' % x for x in tim.recompression_times_ns]}")
        for note in tim.notes:
            print("  note:", note)

    # Thin target -> Case A (release-fan interaction; recompression suppressed).
    _report("h_t=250um", compute_wave_timing(u_fs, al, brass, h_flyer_um=100.0, h_target_um=250.0))
    # Very thin target -> Case B (isolated recompression series at n*P_t).
    _report("h_t=30um", compute_wave_timing(u_fs, al, brass, h_flyer_um=100.0, h_target_um=30.0))
    # No target thickness -> flyer round-trip only.
    _report("no h_target", compute_wave_timing(u_fs, al, brass, h_flyer_um=100.0, h_target_um=None))
