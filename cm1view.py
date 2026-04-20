#!/opt/miniconda3/envs/met/bin/python3
"""cm1view.py — CM1 netCDF output viewer.

Supports:
  • Plan view (x-y at a fixed height level)
  • X-Z cross section (at fixed y)
  • Y-Z cross section (at fixed x)
  • Time stepping with slider, back/forward buttons, and auto-play
  • Wind-vector overlay (u/v on plan view; u/w or v/w on cross-sections)
  • Save current frame as PNG
  • Save a time range as an animated GIF
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import glob
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.colors import Normalize
from matplotlib.animation import PillowWriter
import matplotlib.pyplot as plt

import netCDF4 as nc


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------

class CM1Dataset:
    """Wraps one or more CM1 netCDF files as a unified time series."""

    def __init__(self, paths):
        paths = sorted(paths)
        self._dsets = [nc.Dataset(p, 'r') for p in paths]
        self._build_index()

    def _build_index(self):
        """Build (dset_idx, time_idx_in_dset) for every time step."""
        self._time_map = []   # list of (ds_idx, t_idx)
        self._times    = []   # seconds
        for di, ds in enumerate(self._dsets):
            try:
                t = ds.variables['time'][:]
            except KeyError:
                t = np.array([0.0])
            for ti, tv in enumerate(t):
                self._time_map.append((di, ti))
                self._times.append(float(tv))
        self._times = np.array(self._times)

        # coordinates from first dataset
        ds0 = self._dsets[0]
        self.xh = np.array(ds0.variables['xh'][:])
        self.yh = np.array(ds0.variables['yh'][:])
        try:
            self.zh = np.array(ds0.variables['zh'][:])
        except KeyError:
            nk = next(iter(ds0.dimensions.get(k) for k in ('zh', 'nz') if k in ds0.dimensions), None)
            self.zh = np.arange(len(self.xh)) if nk is None else np.arange(nk)

        # available 3-D and 2-D field names (skip coordinate/metadata vars)
        _coord = {'xh', 'xf', 'yh', 'yf', 'zh', 'zf', 'time', 'f_cor',
                  'ztop', 'umove', 'vmove', 'zs', 'zh_sfc'}
        self.fields_3d = []
        self.fields_2d = []
        for name, var in ds0.variables.items():
            if name in _coord:
                continue
            dims = var.dimensions
            if 'time' not in dims:
                continue
            # spatial dims excluding time
            sp = [d for d in dims if d != 'time']
            if len(sp) == 3:
                self.fields_3d.append(name)
            elif len(sp) == 2:
                self.fields_2d.append(name)
        self.fields_3d.sort()
        self.fields_2d.sort()

    @property
    def ntimes(self):
        return len(self._times)

    @property
    def times(self):
        return self._times

    def get_field(self, name, time_idx):
        di, ti = self._time_map[time_idx]
        ds = self._dsets[di]
        var = ds.variables[name]
        dims = var.dimensions
        t_ax = list(dims).index('time')
        slices = [slice(None)] * len(dims)
        slices[t_ax] = ti
        return np.array(var[tuple(slices)], dtype=float)

    def get_units(self, name):
        ds0 = self._dsets[0]
        var = ds0.variables.get(name)
        if var is None:
            return ''
        return getattr(var, 'units', '') or ''

    def get_longname(self, name):
        ds0 = self._dsets[0]
        var = ds0.variables.get(name)
        if var is None:
            return name
        return getattr(var, 'long_name', name) or name

    def add_paths(self, new_paths):
        """Append new files; returns number successfully added."""
        n_added = 0
        times_list = list(self._times)
        for p in sorted(new_paths):
            try:
                ds = nc.Dataset(p, 'r')
            except Exception:
                continue
            di = len(self._dsets)
            self._dsets.append(ds)
            try:
                t = ds.variables['time'][:]
            except KeyError:
                t = np.array([0.0])
            for ti, tv in enumerate(t):
                self._time_map.append((di, ti))
                times_list.append(float(tv))
            n_added += 1
        self._times = np.array(times_list)
        return n_added

    def close(self):
        for ds in self._dsets:
            try:
                ds.close()
            except Exception:
                pass
        self._dsets.clear()

    def __del__(self):
        self.close()


# ---------------------------------------------------------------------------
# Radar simulation (physics ported from StormBox/simulation/instruments/radar.py,
# adapted for CM1 netCDF multi-species NSSL 2-moment microphysics)
# ---------------------------------------------------------------------------

_RC_LIGHT   = 3e8        # m/s
_RK2        = 0.93       # |K|² liquid water
_RK2_ICE    = 0.20       # |K|² ice/snow
_RE_EFF     = 8.5e6      # m  (standard 4/3 Earth refraction)
_V_RAIN     = 7.0        # m/s  fall speed – rain
_V_SNOW     = 1.5        # m/s  fall speed – snow
_V_GRPL     = 3.0        # m/s  fall speed – graupel
_V_HAIL     = 10.0       # m/s  fall speed – hail
_RL_SYS_DB  = 3.0        # dB  system losses
_RETA       = 0.6        # antenna efficiency
_RS_MIN_W   = 5e-13      # W   minimum detectable signal

# Band: (attenuation a,b), lambda_m, (KDP a,b), gamma
_RBAND = {
    'S':  dict(attn=(0.00017,0.71), lam=0.10, kdp=(0.0365,0.890), gamma=0.053),
    'C':  dict(attn=(0.00080,0.74), lam=0.05, kdp=(0.0743,0.895), gamma=0.250),
    'X':  dict(attn=(0.00372,0.72), lam=0.03, kdp=(0.1507,0.907), gamma=0.370),
    'Ka': dict(attn=(0.026,  0.69), lam=0.008,kdp=(0.300, 0.950), gamma=0.500),
}
_RZDR_A, _RZDR_B = 1.74, -0.66   # ZDR = A*D0 + B (dB)


class CM1Radar:
    """Doppler radar simulator for CM1 netCDF output.

    Adapts StormBox DopplerRadar for CM1's km-based grid and NSSL 2-moment
    microphysics (separate qr / qs / qg / qh species, T from th+prs).
    """

    def __init__(self, radar_x_km, radar_y_km, cfg, ds):
        self._rx = radar_x_km * 1000.0   # m (East from domain origin)
        self._ry = radar_y_km * 1000.0   # m (North)

        # ── Hardware ────────────────────────────────────────────────────────
        self.band    = cfg.get('band', 'S')
        bd           = _RBAND[self.band]
        self.lam_m   = bd['lam']
        self.D_m     = float(cfg.get('dish_m', 4.2))
        self.Pt_W    = float(cfg.get('power_kw', 250.0)) * 1000.0
        self.theta_beam_rad = 1.22 * self.lam_m / self.D_m
        self.theta_beam_deg = np.degrees(self.theta_beam_rad)
        G = _RETA * (np.pi * self.D_m / self.lam_m) ** 2
        Lsys = 10 ** (_RL_SYS_DB / 10.0)
        self._C_radar = (self.Pt_W * G**2 * self.theta_beam_rad**2 *
                         _RC_LIGHT * _RK2 * np.pi**3 /
                         (1024.0 * np.log(2) * self.lam_m**2 * Lsys))

        # ── Operational defaults ─────────────────────────────────────────────
        self.prf_hz         = float(cfg.get('prf_hz', 1000.0))
        self.tau_us         = float(cfg.get('pulse_us', 1.0))
        self.clutter_filter = bool(cfg.get('clutter_filter', False))
        self.v_notch_ms     = float(cfg.get('v_notch_ms', 1.5))

        # ── CM1 grid (km → m) ────────────────────────────────────────────────
        self._xh_m = ds.xh * 1000.0
        self._yh_m = ds.yh * 1000.0
        self._zh_m = ds.zh * 1000.0
        self._nx   = len(ds.xh)
        self._ny   = len(ds.yh)
        self._nz   = len(ds.zh)
        self._dx   = float(self._xh_m[1] - self._xh_m[0]) if self._nx > 1 else 2000.0
        self._dy   = float(self._yh_m[1] - self._yh_m[0]) if self._ny > 1 else 2000.0

        self._fields_3d = set(ds.fields_3d)
        self._update_derived()
        self._precompute_barnes()

    # ── derived quantities ───────────────────────────────────────────────────

    def _update_derived(self):
        tau_s          = self.tau_us * 1e-6
        self.r_max_m   = _RC_LIGHT / (2.0 * self.prf_hz)
        self.v_max_ms  = self.lam_m * self.prf_hz / 4.0
        self.delta_r_m = _RC_LIGHT * tau_s / 2.0
        max_r          = np.hypot(self._nx * self._dx, self._ny * self._dy)
        self.n_gates   = min(max(1, int(self.r_max_m / self.delta_r_m)),
                             max(1, int(max_r / self.delta_r_m)))
        self.n_az      = min(max(1, int(360.0 / self.theta_beam_deg)), 720)
        r_ref = 10e3
        Z_min_lin = _RS_MIN_W * r_ref**2 * 1e18 / (self._C_radar * tau_s)
        self.z_min_ref_dbz = 10.0 * np.log10(max(Z_min_lin, 1e-30))
        # DZ 6.23: M = number of independent samples per dwell (assume 20°/s scan rate)
        self.n_samples = max(1, int(self.prf_hz * self.theta_beam_deg / 20.0))

    # ── Barnes 3×3 horizontal interpolation weights ──────────────────────────

    def _precompute_barnes(self):
        kappa2 = self._dx * self._dx
        az_rad = np.radians(np.linspace(0, 360, self.n_az, endpoint=False))
        gate_r = np.arange(1, self.n_gates + 1, dtype=np.float32) * self.delta_r_m
        gx_m = (self._rx + gate_r[None, :] * np.sin(az_rad[:, None])).ravel().astype(np.float64)
        gy_m = (self._ry + gate_r[None, :] * np.cos(az_rad[:, None])).ravel().astype(np.float64)
        # searchsorted handles stretched/non-uniform grids correctly
        ic0  = np.clip(np.searchsorted(self._xh_m, gx_m) - 1, 0, self._nx - 1).astype(np.int32)
        ir0  = np.clip(np.searchsorted(self._yh_m, gy_m) - 1, 0, self._ny - 1).astype(np.int32)
        di   = np.array([-1,-1,-1, 0,0,0, 1,1,1], dtype=np.int32)
        dj   = np.array([-1, 0, 1,-1,0,1,-1,0,1], dtype=np.int32)
        all_iy = np.clip(ir0[:,None] + di[None,:], 0, self._ny-1).astype(np.int32)
        all_ix = np.clip(ic0[:,None] + dj[None,:], 0, self._nx-1).astype(np.int32)
        # Use actual grid coordinates (not reconstructed from uniform spacing)
        nb_x   = self._xh_m[all_ix]
        nb_y   = self._yh_m[all_iy]
        dxn    = gx_m[:,None] - nb_x
        dyn    = gy_m[:,None] - nb_y
        w      = np.exp(-(dxn*dxn + dyn*dyn).astype(np.float32) / kappa2)
        w     /= np.maximum(w.sum(axis=1, keepdims=True), 1e-10)
        self._biy = all_iy
        self._bix = all_ix
        self._bw  = w.astype(np.float32)

    # ── state extraction from CM1Dataset ─────────────────────────────────────

    def _gf(self, ds, t_idx, names):
        """Return first available 3-D field or zeros."""
        for n in names:
            if n in self._fields_3d:
                return ds.get_field(n, t_idx).astype(np.float32)
        return np.zeros((self._nz, self._ny, self._nx), dtype=np.float32)

    def _extract(self, ds, t_idx):
        th  = self._gf(ds, t_idx, ['th'])
        prs = self._gf(ds, t_idx, ['prs'])
        T_K  = th * (np.maximum(prs, 1.0) / 1e5) ** (287.0 / 1004.0)
        rho  = np.maximum(prs, 1.0) / (287.0 * np.maximum(T_K, 100.0))
        return dict(
            T_K = T_K,
            rho = rho,
            qr  = self._gf(ds, t_idx, ['qr']),
            qc  = self._gf(ds, t_idx, ['qc']),
            qi  = self._gf(ds, t_idx, ['qi']),
            qs  = self._gf(ds, t_idx, ['qs']),
            qg  = self._gf(ds, t_idx, ['qg']),
            qh  = self._gf(ds, t_idx, ['qh']),
            u   = self._gf(ds, t_idx, ['uinterp', 'u']),
            v   = self._gf(ds, t_idx, ['vinterp', 'v']),
            w   = self._gf(ds, t_idx, ['winterp', 'w']),
            tke = self._gf(ds, t_idx, ['tke']),
        )

    # ── multi-species reflectivity (CM1 NSSL 2-moment) ────────────────────────

    def _compute_Z(self, qr, qs, qg, qh, qc, T_K, rho):
        """Z [mm⁶ m⁻³] from CM1's separate hydrometeor species.

        Phase state is derived from species mixing ratios, not temperature.
        Melting-layer Z enhancement scales with liquid/ice coexistence (disorder).
        """
        is_ice = T_K < 273.15

        R_r = np.maximum(qr * rho * _V_RAIN * 3600.0, 0.0)
        Z_r = 200.0 * np.maximum(R_r, 1e-10) ** 1.6

        R_s = np.maximum(qs * rho * _V_SNOW * 3600.0, 0.0)
        Z_s = np.where(is_ice,
                       2000.0 * np.maximum(R_s, 1e-10)**2.0 * (_RK2_ICE / _RK2),
                       2000.0 * np.maximum(R_s, 1e-10)**2.0)

        R_g = np.maximum(qg * rho * _V_GRPL * 3600.0, 0.0)
        Z_g = np.where(is_ice,
                       315.0 * np.maximum(R_g, 1e-10)**1.5 * (_RK2_ICE / _RK2) * 2.5,
                       315.0 * np.maximum(R_g, 1e-10)**1.5)

        R_h = np.maximum(qh * rho * _V_HAIL * 3600.0, 0.0)
        Z_h = np.where(is_ice,
                       630.0 * np.maximum(R_h, 1e-10)**1.75 * (_RK2_ICE / _RK2) * 4.0,
                       630.0 * np.maximum(R_h, 1e-10)**1.75)

        LWC = qc * rho
        Z_c = 1.2e8 * np.maximum(LWC, 0.0) ** 2

        # Melting-layer Z enhancement driven by species coexistence, not temperature.
        # Melting snow develops a liquid shell → dielectric jumps toward liquid.
        # disorder = 4·lf·(1-lf) ∈ [0,1], peaks at 50/50 liquid+ice mix.
        liq_frac = np.maximum(qr, 0.0) / (np.maximum(qr + qs + qg + qh, 0.0) + 1e-15)
        disorder = 4.0 * liq_frac * (1.0 - liq_frac)
        Z_s = Z_s * (1.0 + 3.0 * disorder)   # melting aggregates: up to ×4
        Z_r = Z_r * (1.0 + 1.0 * disorder)   # liquid mixing with melt water

        Z_total = np.maximum(Z_r + Z_s + Z_g + Z_h + Z_c, 1e-10).astype(np.float32)
        R_total = (R_r + R_s + R_g).astype(np.float32)
        return Z_total, R_total, disorder.astype(np.float32)

    def _compute_pol(self, qr, qs, qi, qg, qh, R, Z_true, disorder):
        """ZDR and CC from species mixing ratios — no temperature-based forcing.

        disorder = 4·lf·(1-lf) passed in from _compute_Z (consistent liq_frac).
        """
        q_ice_tot = np.maximum(qs + qi + qg + qh, 1e-15)
        liq_frac  = np.maximum(qr, 0.0) / (np.maximum(qr, 0.0) + q_ice_tot)

        # ZDR (dB)
        D0       = 0.68 * np.maximum(R, 1e-6) ** 0.21
        zdr_rain = np.maximum(_RZDR_A * D0 + _RZDR_B, 0.0)
        # Per-species ice ZDR: snow ~0.25, cloud-ice ~0.30, graupel ~0.10, hail ~0
        zdr_ice  = (qs * 0.25 + qi * 0.30 + qg * 0.10 + qh * 0.0) / q_ice_tot
        # Partial-melt enhancement: large oblate melting aggregates (peaks at ~0.5 mix)
        zdr = liq_frac * zdr_rain + (1.0 - liq_frac) * zdr_ice + 2.0 * disorder

        # CC
        cc_rain = 0.99 - 0.01 * np.clip(R / 50.0, 0.0, 1.0)
        # Per-species CC: snow 0.97, cloud-ice 0.98, graupel 0.96, hail 0.92
        cc_ice  = (qs * 0.97 + qi * 0.98 + qg * 0.96 + qh * 0.92) / q_ice_tot
        cc = liq_frac * cc_rain + (1.0 - liq_frac) * cc_ice - 0.12 * disorder

        no_pcp = Z_true < 1.0
        zdr = np.where(no_pcp, 0.0, zdr).astype(np.float32)
        cc  = np.where(no_pcp, 0.0, np.clip(cc, 0.0, 1.0)).astype(np.float32)
        return zdr, cc

    # ── PPI scan ──────────────────────────────────────────────────────────────

    def scan_ppi(self, ds, t_idx, el_deg):
        """Return (refl, vel, zdr, kdp_r, cc, gate_r_km, az_deg_edges) for one PPI tilt."""
        st = self._extract(ds, t_idx)
        return self._scan_ppi(st, el_deg)

    def _scan_ppi(self, st, el_deg):
        el_rad = np.radians(el_deg)
        az_rad = np.radians(np.linspace(0, 360, self.n_az, endpoint=False)).astype(np.float32)
        gate_r = np.arange(1, self.n_gates + 1, dtype=np.float32) * self.delta_r_m
        N      = self.n_az * self.n_gates

        r_flat  = np.tile(gate_r, self.n_az)
        az_flat = np.repeat(az_rad, self.n_gates)
        sin_az  = np.sin(az_flat)
        cos_az  = np.cos(az_flat)

        # Height AGL with Earth curvature
        h_agl = np.clip(r_flat * np.sin(el_rad) + r_flat**2 / (2 * _RE_EFF),
                        0.0, None).astype(np.float32)

        # Vertical interpolation indices
        Z_LEV   = self._zh_m.astype(np.float32)
        k_lvl   = np.clip(np.searchsorted(Z_LEV, h_agl) - 1, 0, self._nz - 2).astype(np.int32)
        dz_lyr  = np.maximum(Z_LEV[k_lvl + 1] - Z_LEV[k_lvl], 1.0)
        alpha_z = np.clip((h_agl - Z_LEV[k_lvl]) / dz_lyr, 0.0, 1.0).astype(np.float32)

        iy, ix, bw = self._biy, self._bix, self._bw

        def _s3(field):
            k   = k_lvl[:, None]
            lo  = field[k,   iy, ix]
            hi  = field[k+1, iy, ix]
            return ((1.0 - alpha_z[:,None]) * lo + alpha_z[:,None] * hi) * bw
        def _s3sum(field):
            return _s3(field).sum(axis=1).astype(np.float32)

        qr = _s3sum(st['qr']); qc = _s3sum(st['qc']); qi = _s3sum(st['qi'])
        qs = _s3sum(st['qs']); qg = _s3sum(st['qg']); qh = _s3sum(st['qh'])
        T  = _s3sum(st['T_K']); rho = st['rho'][k_lvl, 0, 0]  # profile density
        # Use full Barnes rho
        rho = _s3sum(st['rho'])
        u   = _s3sum(st['u']); v = _s3sum(st['v']); w = _s3sum(st['w'])
        tke = _s3sum(st['tke'])

        Z_true, R, disorder = self._compute_Z(qr, qs, qg, qh, qc, T, rho)

        # Attenuation
        bd = _RBAND[self.band]
        a_att, b_att = bd['attn']
        a_kdp, b_kdp = bd['kdp']
        alpha_dBkm   = (a_att * Z_true**b_att).astype(np.float32)
        A_2d   = (2.0 * alpha_dBkm * (self.delta_r_m / 1000.0)).reshape(self.n_az, self.n_gates)
        A_cum  = np.concatenate([np.zeros((self.n_az, 1), np.float32),
                                 np.cumsum(A_2d, axis=1)[:, :-1]], axis=1).ravel()
        Z_obs_dbz = 10.0 * np.log10(Z_true) - A_cum

        kdp_r    = np.where(R > 0, a_kdp * R**b_kdp, 0.0).astype(np.float32)
        zdr, cc  = self._compute_pol(qr, qs, qi, qg, qh, R, Z_true, disorder)

        # Noise floor: range-dependent MDS; comparing attenuated Z to range-only floor
        # is equivalent to requiring Z_true > Z_min_range + A_cum (attenuation included).
        Z_min_r = self.z_min_ref_dbz + 20.0 * np.log10(np.maximum(r_flat / 10e3, 1e-6))
        below   = Z_obs_dbz < Z_min_r
        Z_obs_dbz[below] = np.nan
        zdr[below]  = np.nan
        kdp_r[below] = np.nan
        cc[below]   = np.nan

        # Radial velocity + DZ 6.23 variance noise + aliasing
        cos_el  = float(np.cos(el_rad))
        sin_el  = float(np.sin(el_rad))
        v_r     = u * sin_az * cos_el + v * cos_az * cos_el + w * sin_el
        sigma_v = np.sqrt(np.maximum(2.0/3.0 * tke, 0.0))
        T_s     = 1.0 / self.prf_hz
        var_v   = sigma_v * self.lam_m / (8.0 * self.n_samples * T_s * np.sqrt(np.pi))
        std_v   = np.sqrt(np.maximum(var_v, 0.0)).astype(np.float32)
        v_r_obs = v_r + np.random.normal(0.0, 1.0, size=v_r.shape).astype(np.float32) * std_v
        v_alias = ((v_r_obs + self.v_max_ms) % (2.0 * self.v_max_ms)) - self.v_max_ms
        v_alias[below] = np.nan

        if self.clutter_filter:
            cf = (np.abs(v_r) < self.v_notch_ms) & ~below
            Z_obs_dbz[cf] = np.nan
            v_alias[cf]   = np.nan
            zdr[cf]  = np.nan
            kdp_r[cf] = np.nan
            cc[cf]   = np.nan

        sh = (self.n_az, self.n_gates)
        az_edges = np.radians(np.linspace(0, 360, self.n_az + 1))
        r_edges  = np.arange(0, self.n_gates + 1, dtype=np.float32) * self.delta_r_m / 1000.0
        vel_2d   = v_alias.reshape(sh)
        _vf      = np.where(np.isfinite(vel_2d), vel_2d, 0.0)
        # Circulation: azimuthal Vr shear × gate area = ΔVr × Δr (r·Δφ cancels)
        circ = (np.gradient(_vf, axis=0) * self.delta_r_m).astype(np.float32)
        circ[~np.isfinite(vel_2d)] = np.nan
        # Radial convergence: -∂Vr/∂r (positive = converging toward radar)
        conv = (-np.gradient(_vf, self.delta_r_m, axis=1)).astype(np.float32)
        conv[~np.isfinite(vel_2d)] = np.nan
        return dict(
            refl   = Z_obs_dbz.reshape(sh),
            vel    = vel_2d,
            zdr    = zdr.reshape(sh),
            kdp    = kdp_r.reshape(sh),
            cc     = cc.reshape(sh),
            circ   = circ,
            conv   = conv,
            az_edges = az_edges,
            r_edges  = r_edges,   # km
            el_deg   = el_deg,
            v_max    = self.v_max_ms,
        )

    # ── RHI scan ──────────────────────────────────────────────────────────────

    def scan_rhi(self, ds, t_idx, az_deg):
        """Return (refl, vel, zdr, kdp, cc, r_km, h_km) arrays for one RHI."""
        st = self._extract(ds, t_idx)
        return self._scan_rhi(st, az_deg)

    def _scan_rhi(self, st, az_deg):
        n_el   = max(2, int(60.0 / max(self.theta_beam_deg, 0.5)))
        el_arr = np.radians(np.linspace(0.5, 60.0, n_el, dtype=np.float32))
        gate_r = np.arange(1, self.n_gates + 1, dtype=np.float32) * self.delta_r_m
        az_rad = float(np.radians(az_deg))
        sin_az = float(np.sin(az_rad))
        cos_az = float(np.cos(az_rad))

        gate_x = (self._rx + gate_r * sin_az).astype(np.float32)
        gate_y = (self._ry + gate_r * cos_az).astype(np.float32)

        # Barnes weights for this fixed azimuth's gates
        kappa2 = self._dx * self._dx
        ic0 = np.clip(np.searchsorted(self._xh_m, gate_x.astype(np.float64)) - 1,
                      0, self._nx - 1).astype(np.int32)
        ir0 = np.clip(np.searchsorted(self._yh_m, gate_y.astype(np.float64)) - 1,
                      0, self._ny - 1).astype(np.int32)
        di  = np.array([-1,-1,-1,0,0,0,1,1,1], dtype=np.int32)
        dj  = np.array([-1,0,1,-1,0,1,-1,0,1], dtype=np.int32)
        riy = np.clip(ir0[:,None] + di[None,:], 0, self._ny-1).astype(np.int32)
        rix = np.clip(ic0[:,None] + dj[None,:], 0, self._nx-1).astype(np.int32)
        nb_x = self._xh_m[rix]
        nb_y = self._yh_m[riy]
        dxn  = gate_x[:,None].astype(np.float64) - nb_x
        dyn  = gate_y[:,None].astype(np.float64) - nb_y
        rw   = np.exp(-(dxn*dxn + dyn*dyn).astype(np.float32) / kappa2)
        rw  /= np.maximum(rw.sum(axis=1, keepdims=True), 1e-10)
        rw   = rw.astype(np.float32)

        r2d   = gate_r[None, :]          # (1, n_g)
        el2d  = el_arr[:, None]          # (n_el, 1)
        h_agl = np.clip(r2d * np.sin(el2d) + r2d**2 / (2 * _RE_EFF),
                        0.0, None).astype(np.float32)

        Z_LEV   = self._zh_m.astype(np.float32)
        k_lvl   = np.clip(np.searchsorted(Z_LEV, h_agl) - 1, 0, self._nz-2).astype(np.int32)
        dz_lyr  = np.maximum(Z_LEV[k_lvl + 1] - Z_LEV[k_lvl], 1.0)
        alpha_z = np.clip((h_agl - Z_LEV[k_lvl]) / dz_lyr, 0.0, 1.0).astype(np.float32)

        iy3 = riy[None, :, :]; ix3 = rix[None, :, :]; w3 = rw[None, :, :]
        k3  = k_lvl[:, :, None]; a3 = alpha_z[:, :, None]

        def _s3r(field):
            lo  = field[k3, iy3, ix3]; hi = field[k3+1, iy3, ix3]
            return ((1.0 - a3)*lo + a3*hi) * w3
        def _s3rs(field):
            return _s3r(field).sum(axis=2).astype(np.float32)

        qr = _s3rs(st['qr']); qc = _s3rs(st['qc']); qi = _s3rs(st['qi'])
        qs = _s3rs(st['qs']); qg = _s3rs(st['qg']); qh = _s3rs(st['qh'])
        T  = _s3rs(st['T_K']); rho = _s3rs(st['rho'])
        u  = _s3rs(st['u']); v = _s3rs(st['v']); w = _s3rs(st['w'])
        tke = _s3rs(st['tke'])

        Z_true, R, disorder = self._compute_Z(qr, qs, qg, qh, qc, T, rho)

        bd = _RBAND[self.band]
        a_att, b_att = bd['attn']
        a_kdp, b_kdp = bd['kdp']
        alpha_dBkm   = (a_att * Z_true**b_att).astype(np.float32)
        A_cum  = np.concatenate([np.zeros((n_el, 1), np.float32),
                                 np.cumsum(2.0 * alpha_dBkm * (self.delta_r_m / 1000.0),
                                           axis=1)[:, :-1]], axis=1)
        Z_obs_dbz = 10.0 * np.log10(Z_true) - A_cum

        kdp_r   = np.where(R > 0, a_kdp * R**b_kdp, 0.0).astype(np.float32)
        zdr, cc = self._compute_pol(qr, qs, qi, qg, qh, R, Z_true, disorder)

        Z_min_r = self.z_min_ref_dbz + 20.0*np.log10(np.maximum(gate_r / 10e3, 1e-6))
        below   = Z_obs_dbz < Z_min_r[None, :]
        Z_obs_dbz[below] = np.nan
        zdr[below]  = np.nan
        kdp_r[below] = np.nan
        cc[below]   = np.nan

        cos_el_2d = np.cos(el2d); sin_el_2d = np.sin(el2d)
        v_r     = u*sin_az*cos_el_2d + v*cos_az*cos_el_2d + w*sin_el_2d
        sigma_v = np.sqrt(np.maximum(2.0/3.0 * tke, 0.0))
        T_s     = 1.0 / self.prf_hz
        var_v   = sigma_v * self.lam_m / (8.0 * self.n_samples * T_s * np.sqrt(np.pi))
        std_v   = np.sqrt(np.maximum(var_v, 0.0)).astype(np.float32)
        v_r_obs = v_r + np.random.normal(0.0, 1.0, size=v_r.shape).astype(np.float32) * std_v
        v_alias = ((v_r_obs + self.v_max_ms) % (2.0*self.v_max_ms)) - self.v_max_ms
        v_alias[below] = np.nan

        if self.clutter_filter:
            cf = (np.abs(v_r) < self.v_notch_ms) & ~below
            Z_obs_dbz[cf] = np.nan; v_alias[cf] = np.nan
            zdr[cf]  = np.nan; kdp_r[cf] = np.nan; cc[cf] = np.nan

        vel_2d = v_alias.astype(np.float32)
        _vf    = np.where(np.isfinite(vel_2d), vel_2d, 0.0)
        # Circulation: elevation Vr shear × gate area = ΔVr × Δr (r·Δel cancels)
        circ = (np.gradient(_vf, axis=0) * self.delta_r_m).astype(np.float32)
        circ[~np.isfinite(vel_2d)] = np.nan
        # Radial convergence: -∂Vr/∂r (positive = converging toward radar)
        conv = (-np.gradient(_vf, self.delta_r_m, axis=1)).astype(np.float32)
        conv[~np.isfinite(vel_2d)] = np.nan
        return dict(
            refl     = Z_obs_dbz.astype(np.float32),
            vel      = vel_2d,
            zdr      = zdr,
            kdp      = kdp_r,
            cc       = cc,
            circ     = circ,
            conv     = conv,
            r_km     = gate_r / 1000.0,
            h_km     = h_agl,   # (n_el, n_gates) in m → convert on display
            az_deg   = az_deg,
            v_max    = self.v_max_ms,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CMAPS = [
    'RdBu_r', 'bwr', 'seismic',
    'viridis', 'plasma', 'inferno', 'magma', 'cividis',
    'Blues', 'Reds', 'Greens', 'BuGn',
    'turbo', 'jet', 'rainbow',
    'Greys', 'binary',
    'gist_ncar', 'gist_rainbow',
]

def _sec_label(s):
    m = s / 60.0
    if m < 60:
        return f"{m:.1f} min"
    h = m / 60.0
    return f"{h:.2f} h"


def _copy_figure_to_clipboard(fig):
    """Render a matplotlib figure as PNG and place it on the system clipboard."""
    import io, sys, tempfile, os, subprocess
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    png_bytes = buf.getvalue()

    if sys.platform == 'darwin':
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            f.write(png_bytes)
            tmp = f.name
        try:
            subprocess.run(
                ['osascript', '-e',
                 f'set the clipboard to (read (POSIX file "{tmp}") as «class PNGf»)'],
                check=True)
        finally:
            os.unlink(tmp)

    elif sys.platform == 'win32':
        try:
            import win32clipboard
            from PIL import Image
            img = Image.open(io.BytesIO(png_bytes)).convert('RGB')
            bmp_buf = io.BytesIO()
            img.save(bmp_buf, 'BMP')
            dib = bmp_buf.getvalue()[14:]   # strip 14-byte BMP file header → DIB
            win32clipboard.OpenClipboard()
            win32clipboard.EmptyClipboard()
            win32clipboard.SetClipboardData(win32clipboard.CF_DIB, dib)
            win32clipboard.CloseClipboard()
        except ImportError:
            raise RuntimeError("Install pywin32 for clipboard image support on Windows.")

    else:   # Linux / other X11
        for cmd in (['xclip', '-selection', 'clipboard', '-t', 'image/png'],
                    ['xsel',  '--clipboard', '--input']):
            try:
                subprocess.run(cmd, input=png_bytes, check=True)
                return
            except (FileNotFoundError, subprocess.CalledProcessError):
                pass
        raise RuntimeError("Install xclip or xsel for clipboard image support on Linux.")


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class CM1Viewer(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("CM1 netCDF Viewer")
        self.minsize(1100, 700)

        self._ds           = None
        self._t_idx        = 0
        self._play_id      = None
        self._playing      = False
        self._last_field   = None
        self._sounding_mode = False
        self._mpl_cid      = None
        self._watch_dir    = None
        self._watch_known  = set()
        self._watch_id     = None

        self._build_vars()
        self._build_ui()

    # ── variables ────────────────────────────────────────────────────────────

    def _build_vars(self):
        self.v_field      = tk.StringVar()
        self.v_cmap       = tk.StringVar(value='RdBu_r')
        self.v_view       = tk.StringVar(value='plan')
        self.v_symcb      = tk.BooleanVar(value=False)
        self.v_vmin       = tk.StringVar(value='')
        self.v_vmax       = tk.StringVar(value='')
        self.v_winds      = tk.BooleanVar(value=False)
        self.v_wind_skip_x = tk.IntVar(value=4)
        self.v_wind_skip_y = tk.IntVar(value=4)
        self.v_wind_type  = tk.StringVar(value='arrows')
        self.v_ctr_field  = tk.StringVar(value='')
        self.v_ctr_levels = tk.IntVar(value=8)
        self.v_ctr_lw     = tk.DoubleVar(value=1.2)
        self.v_ctr_color  = tk.StringVar(value='black')
        self.v_ctr_labels = tk.BooleanVar(value=False)
        self.v_ctr_style  = tk.StringVar(value='solid')
        self.v_ctr_min    = tk.StringVar(value='')
        self.v_ctr_max    = tk.StringVar(value='')
        self.v_ctr_sym    = tk.BooleanVar(value=False)
        self.v_xmin = tk.StringVar(value='')
        self.v_xmax = tk.StringVar(value='')
        self.v_ymin = tk.StringVar(value='')
        self.v_ymax = tk.StringVar(value='')
        self.v_xlim_sym = tk.BooleanVar(value=False)
        self.v_ylim_sym = tk.BooleanVar(value=False)
        self.v_gif_t0     = tk.StringVar(value='0')
        self.v_gif_t1     = tk.StringVar(value='')
        self.v_live       = tk.BooleanVar(value=True)
        # radar
        self.v_radar_band     = tk.StringVar(value='S')
        self.v_radar_dish     = tk.DoubleVar(value=4.2)
        self.v_radar_power    = tk.DoubleVar(value=250.0)
        self.v_radar_prf      = tk.DoubleVar(value=1000.0)
        self.v_radar_pulse    = tk.DoubleVar(value=1.0)
        self.v_radar_clutter  = tk.BooleanVar(value=False)
        self.v_radar_el       = tk.DoubleVar(value=0.5)
        self.v_radar_az       = tk.DoubleVar(value=0.0)
        self.v_radar_product  = tk.StringVar(value='refl')
        self._radar_loc       = None   # (x_km, y_km) or None
        self._radar_obj       = None   # CM1Radar instance
        self._radar_mode      = False  # placement click mode
        self._radar_win       = None   # RadarWindow Toplevel

    # ── UI ───────────────────────────────────────────────────────────────────

    @staticmethod
    def _autocomplete(combo, get_full_list, on_commit=None):
        """Let the user type to filter a combobox; commit on Return or selection."""
        def _filter(event=None):
            if event and event.keysym in ('Return', 'KP_Enter'):
                if on_commit:
                    on_commit()
                return
            if event and event.keysym in ('Up', 'Down', 'Escape', 'Tab'):
                return
            typed = combo.get().lower()
            full  = get_full_list()
            combo['values'] = [v for v in full if typed in v.lower()] if typed else full

        combo.configure(state='normal')
        combo.bind('<KeyRelease>', _filter)
        combo.bind('<<ComboboxSelected>>', lambda _: on_commit() if on_commit else None)

    def _build_ui(self):
        # ── top bar: file + field ────────────────────────────────────────
        top = ttk.Frame(self)
        top.pack(fill='x', padx=6, pady=4)

        ttk.Button(top, text="Open file(s)…", command=self._open).pack(side='left', padx=4)
        ttk.Button(top, text="Open dir…", command=self._open_dir).pack(side='left', padx=4)
        self._snd_btn = ttk.Button(top, text="Take Sounding",
                                   command=self._toggle_sounding_mode)
        self._snd_btn.pack(side='left', padx=4)
        self._file_lbl = ttk.Label(top, text="No file loaded.", foreground='gray')
        self._file_lbl.pack(side='left', padx=8)
        self._watch_lbl = ttk.Label(top, text="", foreground='#080')
        self._watch_lbl.pack(side='left', padx=4)

        ttk.Label(top, text="Field:").pack(side='left')
        self._field_cb = ttk.Combobox(top, textvariable=self.v_field, width=18)
        self._field_cb.pack(side='left', padx=4)
        self._all_fields = []
        self._autocomplete(self._field_cb, lambda: self._all_fields, self._plot)

        ttk.Label(top, text="Colormap:").pack(side='left', padx=(12, 2))
        self._cmap_cb = ttk.Combobox(top, textvariable=self.v_cmap, width=14)
        self._cmap_cb.pack(side='left', padx=4)
        self._autocomplete(self._cmap_cb, lambda: CMAPS, self._plot)
        self.v_cmap.trace_add('write', lambda *_: self._plot())

        # ── main area: fixed left panel + expanding right plot ───────────
        main = ttk.Frame(self)
        main.pack(fill='both', expand=True, padx=6, pady=2)

        # Left controls — scrollable panel
        left_outer = ttk.Frame(main, width=248)
        left_outer.pack(side='left', fill='y')
        left_outer.pack_propagate(False)

        _lsb = ttk.Scrollbar(left_outer, orient='vertical')
        _lsb.pack(side='right', fill='y')
        _lcv = tk.Canvas(left_outer, yscrollcommand=_lsb.set,
                         highlightthickness=0, width=230)
        _lcv.pack(side='left', fill='both', expand=True)
        _lsb.config(command=_lcv.yview)

        left = ttk.Frame(_lcv)
        _lcv_win = _lcv.create_window((0, 0), window=left, anchor='nw')

        def _on_left_configure(e):
            _lcv.configure(scrollregion=_lcv.bbox('all'))
        def _on_lcv_resize(e):
            _lcv.itemconfig(_lcv_win, width=e.width)
        left.bind('<Configure>', _on_left_configure)
        _lcv.bind('<Configure>', _on_lcv_resize)

        def _on_mousewheel(e):
            _lcv.yview_scroll(int(-1 * (e.delta / 120)), 'units')
        left_outer.bind_all('<MouseWheel>', _on_mousewheel)

        self._build_controls(left)

        ttk.Separator(main, orient='vertical').pack(side='left', fill='y', padx=2)

        # Right plot — fills all remaining space
        right = ttk.Frame(main)
        right.pack(side='left', fill='both', expand=True)

        self._fig = Figure(dpi=100)
        # Fixed axes positions so colorbar never steals space from the plot
        self._ax  = self._fig.add_axes([0.10, 0.10, 0.74, 0.82])
        self._cax = self._fig.add_axes([0.87, 0.10, 0.03, 0.82])
        self._cbar = None
        self._canvas = FigureCanvasTkAgg(self._fig, master=right)
        self._canvas.get_tk_widget().pack(fill='both', expand=True)
        tb = NavigationToolbar2Tk(self._canvas, right, pack_toolbar=False)
        tb.pack(fill='x')
        self._mpl_cid = self._fig.canvas.mpl_connect(
            'button_press_event', self._on_canvas_click)

        # ── bottom: time controls + save ────────────────────────────────
        bot = ttk.Frame(self)
        bot.pack(fill='x', padx=6, pady=4)
        self._build_time_bar(bot)

    def _build_controls(self, parent):
        P = {"padx": 6, "pady": 3}

        # View
        vf = ttk.LabelFrame(parent, text="View")
        vf.pack(fill='x', padx=6, pady=6)
        for text, val in [("Plan  (x-y)", 'plan'),
                          ("X-Z  (fix y)", 'xz'),
                          ("Y-Z  (fix x)", 'yz')]:
            ttk.Radiobutton(vf, text=text, variable=self.v_view,
                            value=val, command=self._plot).pack(
                anchor='w', padx=8, pady=2)

        # Level / slice position
        lf = ttk.LabelFrame(parent, text="Level / position")
        lf.pack(fill='x', padx=6, pady=4)

        ttk.Label(lf, text="Z level (plan view):").pack(anchor='w', padx=6, pady=2)
        self._z_slider = ttk.Scale(lf, from_=0, to=1, orient='horizontal',
                                   command=self._on_z_change)
        self._z_slider.pack(fill='x', padx=6)
        self._z_lbl = ttk.Label(lf, text="z = — km", font=('Courier', 9))
        self._z_lbl.pack(anchor='w', padx=6)

        ttk.Label(lf, text="Cross-section pos:").pack(anchor='w', padx=6, pady=(8, 2))
        self._cs_slider = ttk.Scale(lf, from_=0, to=1, orient='horizontal',
                                    command=self._on_cs_change)
        self._cs_slider.set(0.5)
        self._cs_slider.pack(fill='x', padx=6)
        self._cs_lbl = ttk.Label(lf, text="pos = —", font=('Courier', 9))
        self._cs_lbl.pack(anchor='w', padx=6)

        # Colorbar range
        cf = ttk.LabelFrame(parent, text="Color range")
        cf.pack(fill='x', padx=6, pady=4)
        ttk.Checkbutton(cf, text="Symmetric (±max)",
                        variable=self.v_symcb,
                        command=self._plot).pack(anchor='w', padx=6, pady=2)
        for lbl, var in [("Min:", self.v_vmin), ("Max:", self.v_vmax)]:
            row = ttk.Frame(cf)
            row.pack(fill='x', padx=6, pady=1)
            ttk.Label(row, text=lbl, width=4).pack(side='left')
            e = ttk.Entry(row, textvariable=var, width=10)
            e.pack(side='left')
            e.bind('<Return>', lambda _: self._plot())
        ttk.Button(cf, text="Reset to auto",
                   command=self._reset_range).pack(padx=6, pady=4)

        # Axis limits
        af = ttk.LabelFrame(parent, text="Axis limits")
        af.pack(fill='x', padx=6, pady=4)

        def _lim_entry(parent_row, v_this, v_other, v_sym, is_max):
            """Entry that mirrors its partner when symmetric is on."""
            e = ttk.Entry(parent_row, textvariable=v_this, width=7)
            def _apply(event=None):
                if v_sym.get():
                    try:
                        val = float(v_this.get())
                        v_other.set(f'{-val:.6g}')
                    except ValueError:
                        pass
                self._plot()
            e.bind('<Return>', _apply)
            e.bind('<FocusOut>', _apply)
            return e

        for axis_lbl, v_min, v_max, v_sym, sym_cmd in [
            ('X', self.v_xmin, self.v_xmax, self.v_xlim_sym, self._on_xlim_sym),
            ('Y', self.v_ymin, self.v_ymax, self.v_ylim_sym, self._on_ylim_sym),
        ]:
            row = ttk.Frame(af)
            row.pack(fill='x', padx=6, pady=2)
            ttk.Label(row, text=axis_lbl, width=2).pack(side='left')
            ttk.Label(row, text="min", width=3).pack(side='left')
            _lim_entry(row, v_min, v_max, v_sym, is_max=False).pack(side='left', padx=(0, 4))
            ttk.Label(row, text="max", width=3).pack(side='left')
            _lim_entry(row, v_max, v_min, v_sym, is_max=True).pack(side='left')

            sym_row = ttk.Frame(af)
            sym_row.pack(fill='x', padx=6, pady=(0, 2))
            ttk.Checkbutton(sym_row, text=f"Symmetric {axis_lbl} (±)",
                            variable=v_sym,
                            command=sym_cmd).pack(side='left')

        ttk.Button(af, text="Reset to auto",
                   command=self._reset_lims).pack(padx=6, pady=(0, 4))

        # Wind overlay
        wf = ttk.LabelFrame(parent, text="Wind overlay")
        wf.pack(fill='x', padx=6, pady=4)
        ttk.Checkbutton(wf, text="Show winds",
                        variable=self.v_winds,
                        command=self._plot).pack(anchor='w', padx=6, pady=2)
        row = ttk.Frame(wf)
        row.pack(fill='x', padx=6, pady=2)
        ttk.Label(row, text="Skip X:").pack(side='left')
        ttk.Spinbox(row, from_=1, to=50, textvariable=self.v_wind_skip_x,
                    width=4, command=self._plot).pack(side='left', padx=4)
        ttk.Label(row, text="Y:").pack(side='left')
        ttk.Spinbox(row, from_=1, to=50, textvariable=self.v_wind_skip_y,
                    width=4, command=self._plot).pack(side='left', padx=4)
        row2 = ttk.Frame(wf)
        row2.pack(fill='x', padx=6, pady=(0, 4))
        ttk.Radiobutton(row2, text="Arrows", variable=self.v_wind_type,
                        value='arrows', command=self._plot).pack(side='left')
        ttk.Radiobutton(row2, text="Barbs", variable=self.v_wind_type,
                        value='barbs', command=self._plot).pack(side='left', padx=6)

        # Contour overlay
        ctrf = ttk.LabelFrame(parent, text="Contour overlay")
        ctrf.pack(fill='x', padx=6, pady=4)
        ttk.Label(ctrf, text="Variable:").pack(anchor='w', padx=6, pady=(4, 0))
        self._ctr_cb = ttk.Combobox(ctrf, textvariable=self.v_ctr_field, width=18)
        self._ctr_cb.pack(fill='x', padx=6, pady=2)
        self._all_ctr_fields = ['']
        self._autocomplete(self._ctr_cb,
                           lambda: self._all_ctr_fields, self._plot)

        row = ttk.Frame(ctrf)
        row.pack(fill='x', padx=6, pady=2)
        ttk.Label(row, text="Levels:").pack(side='left')
        ttk.Spinbox(row, from_=2, to=40, textvariable=self.v_ctr_levels,
                    width=4, command=self._plot).pack(side='left', padx=4)
        ttk.Label(row, text="LW:").pack(side='left', padx=(8, 0))
        ttk.Spinbox(row, from_=0.1, to=10.0, increment=0.1,
                    textvariable=self.v_ctr_lw, width=5,
                    command=self._plot).pack(side='left', padx=4)
        ttk.Checkbutton(ctrf, text="Label contours", variable=self.v_ctr_labels,
                        command=self._plot).pack(anchor='w', padx=6, pady=(0, 2))

        ttk.Label(ctrf, text="Color / colormap:").pack(anchor='w', padx=6)
        _ctr_colors = [
            'black', 'white', 'gray', 'red', 'blue', 'green',
            'orange', 'cyan', 'magenta', 'yellow',
            'viridis', 'plasma', 'RdBu_r', 'bwr', 'Reds', 'Blues',
            'turbo', 'jet', 'gist_ncar',
        ]
        self._ctr_color_cb = ttk.Combobox(
            ctrf, textvariable=self.v_ctr_color, width=18, values=_ctr_colors)
        self._ctr_color_cb.pack(fill='x', padx=6, pady=(2, 4))
        self._autocomplete(self._ctr_color_cb, lambda: _ctr_colors, self._plot)
        self.v_ctr_color.trace_add('write', lambda *_: self._plot())

        ttk.Label(ctrf, text="Line style:").pack(anchor='w', padx=6)
        _style_row = ttk.Frame(ctrf)
        _style_row.pack(fill='x', padx=6, pady=(0, 6))
        ttk.Radiobutton(_style_row, text="Solid", variable=self.v_ctr_style,
                        value='solid', command=self._plot).pack(side='left')
        ttk.Radiobutton(_style_row, text="Dashed", variable=self.v_ctr_style,
                        value='dashed', command=self._plot).pack(side='left', padx=4)
        ttk.Radiobutton(_style_row, text="+solid /−dash", variable=self.v_ctr_style,
                        value='pn', command=self._plot).pack(side='left')

        ttk.Label(ctrf, text="Range (min / max):").pack(anchor='w', padx=6, pady=(4, 0))
        _cr = ttk.Frame(ctrf)
        _cr.pack(fill='x', padx=6, pady=2)
        _lim_entry(_cr, self.v_ctr_min, self.v_ctr_max, self.v_ctr_sym, is_max=False).pack(side='left')
        ttk.Label(_cr, text="/").pack(side='left', padx=4)
        _lim_entry(_cr, self.v_ctr_max, self.v_ctr_min, self.v_ctr_sym, is_max=True).pack(side='left')
        ttk.Checkbutton(ctrf, text="Symmetric", variable=self.v_ctr_sym,
                        command=self._on_ctr_sym).pack(anchor='w', padx=6, pady=(0, 6))

        # Virtual radar
        self._build_radar_panel(parent)

    def _build_radar_panel(self, parent):
        rf = ttk.LabelFrame(parent, text="Virtual Radar")
        rf.pack(fill='x', padx=6, pady=4)

        # Placement
        self._radar_lbl = ttk.Label(rf, text="No radar placed", foreground='gray',
                                    font=('Courier', 9))
        self._radar_lbl.pack(anchor='w', padx=6, pady=(4, 0))
        self._radar_btn = ttk.Button(rf, text="Place Radar (click plan view)",
                                     command=self._toggle_radar_mode)
        self._radar_btn.pack(fill='x', padx=6, pady=2)

        # Hardware
        hw = ttk.LabelFrame(rf, text="Hardware")
        hw.pack(fill='x', padx=6, pady=2)
        r1 = ttk.Frame(hw); r1.pack(fill='x', padx=4, pady=2)
        ttk.Label(r1, text="Band:").pack(side='left')
        for b in ('S', 'C', 'X', 'Ka'):
            ttk.Radiobutton(r1, text=b, variable=self.v_radar_band,
                            value=b).pack(side='left', padx=2)
        r2a = ttk.Frame(hw); r2a.pack(fill='x', padx=4, pady=(2, 0))
        ttk.Label(r2a, text="Dish (m):").pack(side='left')
        ttk.Spinbox(r2a, from_=0.5, to=10.0, increment=0.1,
                    textvariable=self.v_radar_dish, width=6).pack(side='left', padx=4)
        r2b = ttk.Frame(hw); r2b.pack(fill='x', padx=4, pady=(0, 2))
        ttk.Label(r2b, text="Power (kW):").pack(side='left')
        ttk.Spinbox(r2b, from_=1, to=1000, increment=10,
                    textvariable=self.v_radar_power, width=6).pack(side='left', padx=4)

        # Operational
        op = ttk.LabelFrame(rf, text="Operational")
        op.pack(fill='x', padx=6, pady=2)
        r3a = ttk.Frame(op); r3a.pack(fill='x', padx=4, pady=(2, 0))
        ttk.Label(r3a, text="PRF (Hz):").pack(side='left')
        ttk.Spinbox(r3a, from_=100, to=4000, increment=100,
                    textvariable=self.v_radar_prf, width=7).pack(side='left', padx=4)
        r3b = ttk.Frame(op); r3b.pack(fill='x', padx=4, pady=(0, 2))
        ttk.Label(r3b, text="Pulse (µs):").pack(side='left')
        ttk.Spinbox(r3b, from_=0.1, to=100.0, increment=0.5,
                    textvariable=self.v_radar_pulse, width=6).pack(side='left', padx=4)
        ttk.Checkbutton(op, text="Clutter filter", variable=self.v_radar_clutter
                        ).pack(anchor='w', padx=4, pady=(0, 2))

        # Scan
        sc = ttk.LabelFrame(rf, text="Scan")
        sc.pack(fill='x', padx=6, pady=2)
        r4 = ttk.Frame(sc); r4.pack(fill='x', padx=4, pady=2)
        ttk.Label(r4, text="PPI El (°):").pack(side='left')
        ttk.Spinbox(r4, from_=0.1, to=89.0, increment=0.5,
                    textvariable=self.v_radar_el, width=6).pack(side='left', padx=4)
        ttk.Button(r4, text="Scan PPI",
                   command=lambda: self._do_radar_scan('ppi')).pack(side='right', padx=4)
        r5 = ttk.Frame(sc); r5.pack(fill='x', padx=4, pady=2)
        ttk.Label(r5, text="RHI Az (°):").pack(side='left')
        ttk.Spinbox(r5, from_=0.0, to=359.9, increment=1.0,
                    textvariable=self.v_radar_az, width=6).pack(side='left', padx=4)
        ttk.Button(r5, text="Scan RHI",
                   command=lambda: self._do_radar_scan('rhi')).pack(side='right', padx=4)

        # Product
        pr = ttk.LabelFrame(rf, text="Product")
        pr.pack(fill='x', padx=6, pady=(2, 6))
        _prow = ttk.Frame(pr); _prow.pack(fill='x', padx=4, pady=2)
        for pname, plbl in [('refl','Z  (dBZ)'), ('vel','Vr (m/s)'),
                             ('zdr','ZDR (dB)'), ('kdp','KDP (°/km)'), ('cc','CC')]:
            ttk.Radiobutton(_prow, text=plbl, variable=self.v_radar_product,
                            value=pname,
                            command=self._redisplay_radar).pack(anchor='w')

    def _build_time_bar(self, parent):
        # Playback row
        pb = ttk.Frame(parent)
        pb.pack(fill='x')

        ttk.Button(pb, text="◀◀", width=3,
                   command=self._t_first).pack(side='left', padx=2)
        ttk.Button(pb, text="◀",  width=3,
                   command=self._t_prev).pack(side='left', padx=2)
        self._play_btn = ttk.Button(pb, text="▶", width=3,
                                    command=self._toggle_play)
        self._play_btn.pack(side='left', padx=2)
        ttk.Button(pb, text="▶",  width=3,
                   command=self._t_next).pack(side='left', padx=2)
        ttk.Button(pb, text="▶▶", width=3,
                   command=self._t_last).pack(side='left', padx=2)

        self._t_slider = ttk.Scale(pb, from_=0, to=1, orient='horizontal',
                                   command=self._on_t_change)
        self._t_slider.pack(side='left', fill='x', expand=True, padx=8)

        self._t_lbl = ttk.Label(pb, text="t = —", width=14,
                                 font=('Courier', 10, 'bold'))
        self._t_lbl.pack(side='left', padx=4)

        # Speed
        ttk.Label(pb, text="Speed (fps):").pack(side='left', padx=(12, 2))
        self.v_fps = tk.IntVar(value=4)
        ttk.Spinbox(pb, from_=1, to=30, textvariable=self.v_fps,
                    width=4).pack(side='left')

        ttk.Checkbutton(pb, text="Live", variable=self.v_live).pack(
            side='left', padx=(12, 2))

        # Save row
        sv = ttk.Frame(parent)
        sv.pack(fill='x', pady=(4, 0))

        ttk.Button(sv, text="Save PNG",
                   command=self._save_png).pack(side='left', padx=4)
        ttk.Button(sv, text="Copy PNG",
                   command=self._copy_to_clipboard).pack(side='left', padx=4)

        ttk.Label(sv, text="  Save GIF  t₀ (s):").pack(side='left')
        ttk.Entry(sv, textvariable=self.v_gif_t0, width=8).pack(side='left', padx=2)
        ttk.Label(sv, text="t₁ (s):").pack(side='left', padx=(4, 0))
        ttk.Entry(sv, textvariable=self.v_gif_t1, width=8).pack(side='left', padx=2)
        ttk.Button(sv, text="Save GIF",
                   command=self._save_gif).pack(side='left', padx=4)
        self._gif_progress = ttk.Label(sv, text="", foreground='#226')
        self._gif_progress.pack(side='left', padx=4)

    # ── radar ─────────────────────────────────────────────────────────────────

    def _toggle_radar_mode(self):
        if self._ds is None:
            messagebox.showwarning("No data", "Open a file first.")
            return
        self._radar_mode = not self._radar_mode
        if self._radar_mode:
            self._radar_btn.config(text="Cancel placement")
            self._canvas.get_tk_widget().config(cursor='crosshair')
            if self._mpl_cid is None:
                self._mpl_cid = self._canvas.mpl_connect(
                    'button_press_event', self._on_canvas_click)
        else:
            self._radar_btn.config(text="Place Radar (click plan view)")
            self._canvas.get_tk_widget().config(cursor='')

    def _on_canvas_click(self, event):
        if event.inaxes != self._ax or event.button != 1:
            return
        if self._sounding_mode:
            self._sounding_mode = False
            self._snd_btn.config(text="Take Sounding")
            self._canvas.get_tk_widget().config(cursor='')
            self._take_sounding(event.xdata, event.ydata)
        elif self._radar_mode:
            self._radar_mode = False
            self._radar_btn.config(text="Place Radar (click plan view)")
            self._canvas.get_tk_widget().config(cursor='')
            if self._ds is None or event.xdata is None or event.ydata is None:
                return
            self._radar_loc = (event.xdata, event.ydata)
            self._radar_lbl.config(
                text=f"Radar @ ({event.xdata:.1f}, {event.ydata:.1f}) km",
                foreground='black')
            self._radar_obj = None

    def _build_radar_obj(self):
        """(Re)create CM1Radar if location or config changed."""
        if self._radar_loc is None or self._ds is None:
            return None
        cfg = dict(
            band        = self.v_radar_band.get(),
            dish_m      = self.v_radar_dish.get(),
            power_kw    = self.v_radar_power.get(),
            prf_hz      = self.v_radar_prf.get(),
            pulse_us    = self.v_radar_pulse.get(),
            clutter_filter = self.v_radar_clutter.get(),
        )
        x_km, y_km = self._radar_loc
        return CM1Radar(x_km, y_km, cfg, self._ds)

    def _do_radar_scan(self, mode):
        if self._ds is None:
            messagebox.showwarning("No data", "Open a file first.")
            return
        if self._radar_loc is None:
            messagebox.showwarning("No radar", "Place a radar first (click plan view).")
            return
        radar = self._build_radar_obj()
        if radar is None:
            return
        self._radar_obj = radar
        scan_angle = self.v_radar_el.get() if mode == 'ppi' else self.v_radar_az.get()

        import threading
        self._gif_progress.config(text=f"Scanning {mode.upper()}…")
        self.update_idletasks()

        def _run():
            try:
                if mode == 'ppi':
                    result = radar.scan_ppi(self._ds, self._t_idx, scan_angle)
                else:
                    result = radar.scan_rhi(self._ds, self._t_idx, scan_angle)
                self.after(0, lambda: self._show_radar(result, mode, radar, scan_angle))
            except Exception as e:
                self.after(0, lambda e=e: messagebox.showerror("Radar error", str(e)))
            finally:
                self.after(0, lambda: self._gif_progress.config(text=""))

        threading.Thread(target=_run, daemon=True).start()

    def _show_radar(self, result, mode, radar, scan_angle):
        """Open or update the RadarWindow with the latest scan."""
        if self._radar_win is None or not self._radar_win.winfo_exists():
            self._radar_win = RadarWindow(self)
        self._radar_win.update_scan(result, mode, self.v_radar_product.get(),
                                    self._radar_loc,
                                    ds=self._ds, radar=radar,
                                    scan_angle=scan_angle, t_idx=self._t_idx)

    def _redisplay_radar(self):
        """Re-render current scan with new product selection."""
        if self._radar_win is not None and self._radar_win.winfo_exists():
            self._radar_win.replot(self.v_radar_product.get())

    # ── sounding ─────────────────────────────────────────────────────────────

    def _toggle_sounding_mode(self):
        if self._ds is None:
            messagebox.showwarning("No data", "Open a file first.")
            return
        self._sounding_mode = not self._sounding_mode
        if self._sounding_mode:
            self._radar_mode = False   # cancel radar placement if active
            self._radar_btn.config(text="Place Radar (click plan view)")
            self._snd_btn.config(text="Cancel Sounding")
            self._canvas.get_tk_widget().config(cursor='crosshair')
        else:
            self._snd_btn.config(text="Take Sounding")
            self._canvas.get_tk_widget().config(cursor='')

    def _take_sounding(self, cx, cy):
        try:
            import sounderpy as spy
        except ImportError:
            messagebox.showerror("sounderpy missing",
                                 "Install sounderpy:  pip install sounderpy")
            return

        ds  = self._ds
        t   = self._t_idx
        xh, yh, zh = ds.xh, ds.yh, ds.zh
        view = self.v_view.get()
        frac = float(self._cs_slider.get()) if hasattr(self, '_cs_slider') else 0.5

        # Resolve clicked coordinates to grid indices
        if view == 'plan':
            xi = int(np.argmin(np.abs(xh - cx)))
            yi = int(np.argmin(np.abs(yh - cy)))
        elif view == 'xz':
            xi = int(np.argmin(np.abs(xh - cx)))
            yi = int(np.clip(frac * (len(yh) - 1), 0, len(yh) - 1))
        else:  # yz
            xi = int(np.clip(frac * (len(xh) - 1), 0, len(xh) - 1))
            yi = int(np.argmin(np.abs(yh - cx)))

        def _col(name):
            if name in ds.fields_3d:
                return ds.get_field(name, t)[:, yi, xi]
            return None

        th_col  = _col('th')
        prs_col = _col('prs')
        qv_col  = _col('qv')
        u_col   = _col('uinterp') if 'uinterp' in ds.fields_3d else _col('u')
        v_col   = _col('vinterp') if 'vinterp' in ds.fields_3d else _col('v')

        if th_col is None or prs_col is None:
            messagebox.showerror("Sounding error",
                                 "Need 'th' and 'prs' output fields to build a sounding.")
            return

        Rd, cp = 287.04, 1004.0
        T_c  = th_col * (prs_col / 1e5) ** (Rd / cp) - 273.15

        if qv_col is not None:
            qv_c  = np.maximum(qv_col, 1e-10)
            e_pa  = qv_c * prs_col / (0.622 + qv_c)
            e_pa  = np.maximum(e_pa, 1e-3)
            Td_c  = 243.5 * np.log(e_pa / 611.2) / (17.67 - np.log(e_pa / 611.2))
        else:
            Td_c  = T_c - 30.0

        from metpy.units import units as munits

        # zh in km → m
        z_m   = zh * 1000.0 if np.nanmax(zh) < 1000 else zh
        p_hpa = prs_col / 100.0
        u_ms  = u_col if u_col is not None else np.zeros_like(T_c)
        v_ms  = v_col if v_col is not None else np.zeros_like(T_c)

        clean_data = {
            'p':  p_hpa * munits('hPa'),
            'T':  T_c   * munits('degC'),
            'Td': Td_c  * munits('degC'),
            'u':  u_ms  * munits('m/s'),
            'v':  v_ms  * munits('m/s'),
            'z':  z_m   * munits('m'),
        }
        t_str = _sec_label(ds.times[t])
        loc_str = f'({xh[xi]:.1f}, {yh[yi]:.1f}) km'
        clean_data['site_info'] = {
            'site-name':   f'CM1  {loc_str}',
            'site-latlon': (0.0, 0.0),
            'site-elv':    0,
            'source':      'CM1',
            'model':       'CM1',
            'fcst-hour':   0,
            'run-time':    ['', '', '', ''],
            'valid-time':  ['', '', '', ''],
        }
        clean_data['titles'] = {
            'top_title':   f'CM1 Model Sounding  |  {loc_str}  |  t = {t_str}',
            'left_title':  'CM1',
            'right_title': '',
        }
        import matplotlib
        import matplotlib.pyplot as mplt
        import tempfile, subprocess, sys

        before = set(mplt.get_fignums())
        _orig_show = mplt.show
        mplt.show = lambda *a, **kw: None
        try:
            spy.build_sounding(clean_data, style='full', color_blind=False,
                               dark_mode=False)
        except Exception as e:
            mplt.show = _orig_show
            messagebox.showerror("sounderpy error", str(e))
            return
        finally:
            mplt.show = _orig_show

        new_figs = set(mplt.get_fignums()) - before
        if not new_figs:
            return
        fn = next(iter(new_figs))
        fig = mplt.figure(fn)
        tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        fig.savefig(tmp.name, dpi=150, bbox_inches='tight')
        mplt.close(fig)
        tmp.close()
        if sys.platform == 'darwin':
            subprocess.Popen(['open', tmp.name])
        elif sys.platform.startswith('linux'):
            subprocess.Popen(['xdg-open', tmp.name])
        else:
            subprocess.Popen(['start', tmp.name], shell=True)

    # ── open ─────────────────────────────────────────────────────────────────

    def _open(self):
        paths = filedialog.askopenfilenames(
            title="Open CM1 netCDF file(s)",
            filetypes=[("NetCDF files", "*.nc"), ("All files", "*.*")])
        if not paths:
            return
        try:
            new_ds = CM1Dataset(list(paths))
        except Exception as e:
            messagebox.showerror("Load error", str(e))
            return
        self._stop_watching()
        names = [os.path.basename(p) for p in paths]
        label = (f"{len(names)} file(s): {', '.join(names[:3])}"
                 + ("…" if len(names) > 3 else ""))
        self._apply_dataset(new_ds, label)

    def _open_dir(self):
        directory = filedialog.askdirectory(title="Open CM1 output directory")
        if not directory:
            return
        paths = sorted(glob.glob(os.path.join(directory, '*.nc')))
        if not paths:
            messagebox.showwarning("No files", "No .nc files found in that directory.")
            return
        try:
            new_ds = CM1Dataset(paths)
        except Exception as e:
            messagebox.showerror("Load error", str(e))
            return
        self._stop_watching()
        self._apply_dataset(new_ds, f"{os.path.basename(directory)}/ ({len(paths)} files)")
        self._start_watching(directory, set(paths))

    def _apply_dataset(self, new_ds, label_text):
        if self._ds:
            self._ds.close()
        self._ds = new_ds

        self._file_lbl.config(text=label_text, foreground='black')

        all_fields = self._ds.fields_3d + self._ds.fields_2d
        self._all_fields = all_fields
        self._all_ctr_fields = [''] + all_fields
        self._field_cb['values'] = all_fields
        self._ctr_cb['values']   = self._all_ctr_fields
        self._cmap_cb['values']  = CMAPS
        if all_fields and not self.v_field.get():
            self.v_field.set(all_fields[0])
        if self.v_ctr_field.get() not in all_fields:
            self.v_ctr_field.set('')

        nt = self._ds.ntimes
        self._t_slider.config(to=max(nt - 1, 1))
        self._t_slider.set(0)
        self._t_idx = 0

        nk = len(self._ds.zh)
        cur_ki = int(self._z_slider.get())
        self._z_slider.config(to=max(nk - 1, 1))
        self._z_slider.set(min(cur_ki, nk - 1))

        self.v_gif_t1.set(str(int(self._ds.times[-1])) if nt > 0 else '')
        self._plot()
        self._canvas.draw()

    def _start_watching(self, directory, known_paths):
        self._watch_dir   = directory
        self._watch_known = known_paths
        self._watch_lbl.config(text=f"⏺ Watching {os.path.basename(directory)}/")
        self._schedule_poll()

    def _stop_watching(self):
        if self._watch_id is not None:
            self.after_cancel(self._watch_id)
            self._watch_id = None
        self._watch_dir = None
        self._watch_known.clear()
        self._watch_lbl.config(text="")

    def _schedule_poll(self):
        self._watch_id = self.after(5000, self._poll_dir)

    def _poll_dir(self):
        if self._watch_dir is None or self._ds is None:
            return
        current = set(glob.glob(os.path.join(self._watch_dir, '*.nc')))
        new_paths = sorted(current - self._watch_known)
        if new_paths:
            n = self._ds.add_paths(new_paths)
            if n:
                self._watch_known |= set(new_paths)
                nt = self._ds.ntimes
                was_at_end = (self._t_idx >= nt - n - 1)
                self._t_slider.config(to=max(nt - 1, 1))
                self.v_gif_t1.set(str(int(self._ds.times[-1])))
                name = os.path.basename(self._watch_dir)
                self._watch_lbl.config(
                    text=f"⏺ Watching {name}/ ({len(self._watch_known)} files)")
                if self.v_live.get() and (was_at_end or not self._playing):
                    self._set_time(nt - 1)
        self._schedule_poll()

    # ── slider callbacks ─────────────────────────────────────────────────────

    def _on_t_change(self, val):
        if self._ds is None:
            return
        idx = min(int(float(val)), self._ds.ntimes - 1)
        if idx != self._t_idx:
            self._t_idx = idx
            self._plot()

    def _on_z_change(self, val):
        if self._ds is None:
            return
        self._update_z_label()
        self._plot()

    def _on_cs_change(self, val):
        if self._ds is None:
            return
        self._update_cs_label()
        self._plot()

    def _update_z_label(self):
        if self._ds is None:
            return
        ki = int(self._z_slider.get())
        ki = np.clip(ki, 0, len(self._ds.zh) - 1)
        self._z_lbl.config(text=f"z = {self._ds.zh[ki]:.3f} km")

    def _update_cs_label(self):
        if self._ds is None:
            return
        view = self.v_view.get()
        frac = float(self._cs_slider.get())
        if view == 'xz':
            coords = self._ds.yh
            val = coords[0] + frac * (coords[-1] - coords[0])
            self._cs_lbl.config(text=f"y = {val:.2f} km")
        elif view == 'yz':
            coords = self._ds.xh
            val = coords[0] + frac * (coords[-1] - coords[0])
            self._cs_lbl.config(text=f"x = {val:.2f} km")
        else:
            self._cs_lbl.config(text="(used in cross sections)")

    # ── time controls ────────────────────────────────────────────────────────

    def _t_first(self):
        self._set_time(0)

    def _t_last(self):
        if self._ds:
            self._set_time(self._ds.ntimes - 1)

    def _t_prev(self):
        self._set_time(max(0, self._t_idx - 1))

    def _t_next(self):
        if self._ds:
            self._set_time(min(self._ds.ntimes - 1, self._t_idx + 1))

    def _set_time(self, idx):
        self._t_idx = idx
        self._t_slider.set(idx)
        self._plot()

    def _toggle_play(self):
        if self._playing:
            self._playing = False
            if self._play_id is not None:
                self.after_cancel(self._play_id)
                self._play_id = None
            self._play_btn.config(text="▶")
        else:
            self._playing = True
            self._play_btn.config(text="⏸")
            self._advance_play()

    def _advance_play(self):
        if not self._playing or self._ds is None:
            return
        self._t_next()
        if self._t_idx >= self._ds.ntimes - 1:
            self._playing = False
            self._play_id = None
            self._play_btn.config(text="▶")
            return
        fps = max(1, self.v_fps.get())
        self._play_id = self.after(int(1000 / fps), self._advance_play)

    # ── colorbar range ───────────────────────────────────────────────────────

    def _reset_range(self):
        self.v_vmin.set('')
        self.v_vmax.set('')
        self._plot()

    def _reset_lims(self):
        for v in (self.v_xmin, self.v_xmax, self.v_ymin, self.v_ymax):
            v.set('')
        self.v_xlim_sym.set(False)
        self.v_ylim_sym.set(False)
        self._plot()

    def _on_xlim_sym(self):
        if self.v_xlim_sym.get():
            # Mirror whichever bound is already set
            try:
                self.v_xmin.set(f'{-float(self.v_xmax.get()):.6g}')
            except ValueError:
                try:
                    self.v_xmax.set(f'{-float(self.v_xmin.get()):.6g}')
                except ValueError:
                    pass
        self._plot()

    def _on_ylim_sym(self):
        if self.v_ylim_sym.get():
            try:
                self.v_ymin.set(f'{-float(self.v_ymax.get()):.6g}')
            except ValueError:
                try:
                    self.v_ymax.set(f'{-float(self.v_ymin.get()):.6g}')
                except ValueError:
                    pass
        self._plot()

    def _on_ctr_sym(self):
        if self.v_ctr_sym.get():
            try:    lo = float(self.v_ctr_min.get())
            except ValueError: lo = None
            try:    hi = float(self.v_ctr_max.get())
            except ValueError: hi = None
            if lo is not None or hi is not None:
                amax = max(abs(v) for v in [lo, hi] if v is not None)
                self.v_ctr_min.set(f'{-amax:.6g}')
                self.v_ctr_max.set(f'{amax:.6g}')
        self._plot()

    def _get_lims(self, v_min, v_max, v_sym, auto_data):
        try:
            lo = float(v_min.get())
        except ValueError:
            lo = None
        try:
            hi = float(v_max.get())
        except ValueError:
            hi = None
        if lo is None and hi is None:
            return None, None   # fully auto
        if lo is None:
            lo = -hi if v_sym.get() else float(np.nanmin(auto_data))
        if hi is None:
            hi = -lo if v_sym.get() else float(np.nanmax(auto_data))
        return lo, hi

    def _get_range(self, data):
        try:
            vmin = float(self.v_vmin.get())
        except ValueError:
            vmin = None
        try:
            vmax = float(self.v_vmax.get())
        except ValueError:
            vmax = None
        if vmin is None or vmax is None:
            finite = data[np.isfinite(data)]
            dmin = float(finite.min()) if len(finite) else 0.0
            dmax = float(finite.max()) if len(finite) else 1.0
            if vmin is None:
                vmin = dmin
            if vmax is None:
                vmax = dmax
        if self.v_symcb.get():
            amax = max(abs(vmin), abs(vmax))
            vmin, vmax = -amax, amax
        return vmin, vmax

    # ── plot ─────────────────────────────────────────────────────────────────

    def _plot(self):
        if self._ds is None:
            return
        field = self.v_field.get()
        if not field:
            return

        try:
            data = self._ds.get_field(field, self._t_idx)
        except Exception as e:
            self._ax.clear()
            self._ax.text(0.5, 0.5, f"Error reading field:\n{e}",
                          ha='center', va='center',
                          transform=self._ax.transAxes, color='red')
            self._canvas.draw_idle()
            return

        t_sec  = self._ds.times[self._t_idx]
        t_str  = _sec_label(t_sec)
        self._t_lbl.config(text=f"t = {t_str}")
        self._t_slider.set(self._t_idx)
        self._update_z_label()
        self._update_cs_label()

        view   = self.v_view.get()
        cmap   = self.v_cmap.get()
        xh     = self._ds.xh
        yh     = self._ds.yh
        zh     = self._ds.zh

        ax = self._ax
        ax.clear()

        is_3d = data.ndim == 3

        if not is_3d:
            # 2-D surface field: data is (ny, nx)
            try:
                plot_data = data
                vmin, vmax = self._get_range(plot_data)
                im = ax.pcolormesh(xh, yh, plot_data,
                                   cmap=cmap, norm=Normalize(vmin, vmax),
                                   shading='nearest')
                self._add_colorbar(im, field)
                ax.set_xlabel('x  (km)')
                ax.set_ylabel('y  (km)')
                ax.set_title(
                    f"{field}  —  t = {t_str}  [{self._ds.get_units(field)}]",
                    fontsize=10)
                xlim = self._get_lims(self.v_xmin, self.v_xmax, self.v_xlim_sym,
                                      np.array(ax.get_xlim()))
                ylim = self._get_lims(self.v_ymin, self.v_ymax, self.v_ylim_sym,
                                      np.array(ax.get_ylim()))
                if xlim[0] is not None:
                    ax.set_xlim(xlim)
                if ylim[0] is not None:
                    ax.set_ylim(ylim)
            except Exception as e:
                ax.text(0.5, 0.5, f"Error plotting 2D field:\n{e}",
                        ha='center', va='center',
                        transform=ax.transAxes, color='red')
            self._canvas.draw_idle()
            return

        # 3-D field
        ki  = int(np.clip(self._z_slider.get(), 0, len(zh) - 1))
        frac = float(self._cs_slider.get())

        if view == 'plan':
            # data is (nz, ny, nx); select level ki → (ny, nx)
            plot_data = data[ki, :, :]
            vmin, vmax = self._get_range(plot_data)
            im = ax.pcolormesh(xh, yh, plot_data,
                               cmap=cmap, norm=Normalize(vmin, vmax),
                               shading='nearest')
            self._add_colorbar(im, field)
            if self.v_winds.get():
                self._overlay_winds_plan(ax, ki, xh, yh, t_sec)
            ax.set_xlabel('x  (km)')
            ax.set_ylabel('y  (km)')
            ax.set_title(
                f"{field}  —  z = {zh[ki]:.2f} km,  t = {t_str}"
                f"  [{self._ds.get_units(field)}]",
                fontsize=10)

        elif view == 'xz':
            ji = int(np.clip(frac * (len(yh) - 1), 0, len(yh) - 1))
            # data is (nz, ny, nx); select y=ji → (nz, nx)
            plot_data = data[:, ji, :]
            vmin, vmax = self._get_range(plot_data)
            im = ax.pcolormesh(xh, zh, plot_data,
                               cmap=cmap, norm=Normalize(vmin, vmax),
                               shading='nearest')
            self._add_colorbar(im, field)
            if self.v_winds.get():
                self._overlay_winds_xz(ax, ji, xh, zh, t_sec)
            ax.set_xlabel('x  (km)')
            ax.set_ylabel('z  (km)')
            ax.set_title(
                f"{field}  —  y = {yh[ji]:.2f} km,  t = {t_str}"
                f"  [{self._ds.get_units(field)}]",
                fontsize=10)

        elif view == 'yz':
            ii = int(np.clip(frac * (len(xh) - 1), 0, len(xh) - 1))
            # data is (nz, ny, nx); select x=ii → (nz, ny)
            plot_data = data[:, :, ii]
            vmin, vmax = self._get_range(plot_data)
            im = ax.pcolormesh(yh, zh, plot_data,
                               cmap=cmap, norm=Normalize(vmin, vmax),
                               shading='nearest')
            self._add_colorbar(im, field)
            if self.v_winds.get():
                self._overlay_winds_yz(ax, ii, yh, zh, t_sec)
            ax.set_xlabel('y  (km)')
            ax.set_ylabel('z  (km)')
            ax.set_title(
                f"{field}  —  x = {xh[ii]:.2f} km,  t = {t_str}"
                f"  [{self._ds.get_units(field)}]",
                fontsize=10)

        ax.set_aspect('auto')
        self._draw_contour(ax, view, ki, frac, xh, yh, zh)

        # Apply user axis limits (x = horizontal axis, y = vertical axis)
        xlim = self._get_lims(self.v_xmin, self.v_xmax, self.v_xlim_sym,
                               np.array(ax.get_xlim()))
        ylim = self._get_lims(self.v_ymin, self.v_ymax, self.v_ylim_sym,
                               np.array(ax.get_ylim()))
        if xlim[0] is not None:
            ax.set_xlim(xlim)
        if ylim[0] is not None:
            ax.set_ylim(ylim)

        self._canvas.draw_idle()

    def _draw_contour(self, ax, view, ki, frac, xh, yh, zh):
        ctr_field = self.v_ctr_field.get()
        if not ctr_field:
            return
        try:
            cdata = self._ds.get_field(ctr_field, self._t_idx)
        except Exception:
            return

        import matplotlib.cm as mpl_cm
        color_val = self.v_ctr_color.get()
        nlevs     = max(2, self.v_ctr_levels.get())
        is_cmap   = color_val in plt.colormaps()

        # slice same way as pcolormesh
        if cdata.ndim == 2:
            if view == 'plan':
                cslice, X, Y = cdata, xh, yh
            elif view == 'xz':
                ji = int(np.clip(frac * (len(yh) - 1), 0, len(yh) - 1))
                cslice, X, Y = cdata, xh, zh
            else:
                ii = int(np.clip(frac * (len(xh) - 1), 0, len(xh) - 1))
                cslice, X, Y = cdata, yh, zh
        else:
            if view == 'plan':
                cslice, X, Y = cdata[ki, :, :], xh, yh
            elif view == 'xz':
                ji = int(np.clip(frac * (len(yh) - 1), 0, len(yh) - 1))
                cslice, X, Y = cdata[:, ji, :], xh, zh
            else:
                ii = int(np.clip(frac * (len(xh) - 1), 0, len(xh) - 1))
                cslice, X, Y = cdata[:, :, ii], yh, zh

        try:
            style    = self.v_ctr_style.get()
            label    = self.v_ctr_labels.get()
            lw       = max(0.1, self.v_ctr_lw.get())
            color_kw = {'cmap': color_val} if is_cmap else {'colors': color_val}

            # resolve explicit min/max/sym
            try:    clo = float(self.v_ctr_min.get())
            except ValueError: clo = None
            try:    chi = float(self.v_ctr_max.get())
            except ValueError: chi = None
            sym = self.v_ctr_sym.get()
            if clo is None and chi is None:
                clo = float(np.nanmin(cslice))
                chi = float(np.nanmax(cslice))
            else:
                clo = clo if clo is not None else float(np.nanmin(cslice))
                chi = chi if chi is not None else float(np.nanmax(cslice))
            if sym:
                amax = max(abs(clo), abs(chi))
                clo, chi = -amax, amax
            levels = np.linspace(clo, chi, nlevs)

            if style == 'pn':
                pos = [l for l in levels if l > 0]
                neg = [l for l in levels if l < 0]
                for levs, ls in [(pos, 'solid'), (neg, 'dashed')]:
                    if not levs:
                        continue
                    cs = ax.contour(X, Y, cslice, levels=levs,
                                    linewidths=lw, linestyles=ls, **color_kw)
                    if label:
                        ax.clabel(cs, inline=True, fontsize=7, fmt='%g')
            else:
                cs = ax.contour(X, Y, cslice, levels=levels, linewidths=lw,
                                linestyles=style, **color_kw)
                if label:
                    ax.clabel(cs, inline=True, fontsize=7, fmt='%g')
        except Exception:
            pass

    def _add_colorbar(self, im, field):
        if self._cbar is None:
            self._cbar = self._fig.colorbar(im, cax=self._cax)
        else:
            self._cbar.update_normal(im)
        units = self._ds.get_units(field) or ''
        self._cbar.set_label(units, fontsize=9)

    # ── wind overlays ────────────────────────────────────────────────────────

    def _wind_vars(self, view):
        """Return (u_name, v_name) appropriate for the view."""
        if view == 'plan':
            return 'uinterp', 'vinterp'
        elif view == 'xz':
            return 'uinterp', 'winterp'
        else:
            return 'vinterp', 'winterp'

    def _draw_wind(self, ax, x1d, y1d, u, v):
        """Draw quiver arrows or barbs depending on user selection."""
        if self.v_wind_type.get() == 'barbs':
            ax.barbs(x1d, y1d, u, v, color='k', alpha=0.7,
                     length=6, linewidth=0.8)
        else:
            ax.quiver(x1d, y1d, u, v,
                      scale=None, color='k', alpha=0.6, width=0.002)

    def _overlay_winds_plan(self, ax, ki, xh, yh, t_sec):
        # wind data is (nz, ny, nx); select level ki → (ny, nx)
        sx = max(1, self.v_wind_skip_x.get())
        sy = max(1, self.v_wind_skip_y.get())
        for u_n, v_n in [('uinterp', 'vinterp'), ('u', 'v')]:
            if u_n in self._ds.fields_3d and v_n in self._ds.fields_3d:
                u = self._ds.get_field(u_n, self._t_idx)[ki, ::sy, ::sx]
                v = self._ds.get_field(v_n, self._t_idx)[ki, ::sy, ::sx]
                self._draw_wind(ax, xh[::sx], yh[::sy], u, v)
                return

    def _overlay_winds_xz(self, ax, ji, xh, zh, t_sec):
        # wind data is (nz, ny, nx); select y=ji → (nz, nx)
        sx = max(1, self.v_wind_skip_x.get())
        sy = max(1, self.v_wind_skip_y.get())
        for u_n in ['uinterp', 'u']:
            for w_n in ['winterp', 'w']:
                if u_n in self._ds.fields_3d and w_n in self._ds.fields_3d:
                    u = self._ds.get_field(u_n, self._t_idx)[::sy, ji, ::sx]
                    w = self._ds.get_field(w_n, self._t_idx)[::sy, ji, ::sx]
                    self._draw_wind(ax, xh[::sx], zh[::sy], u, w)
                    return

    def _overlay_winds_yz(self, ax, ii, yh, zh, t_sec):
        # wind data is (nz, ny, nx); select x=ii → (nz, ny)
        sx = max(1, self.v_wind_skip_x.get())
        sy = max(1, self.v_wind_skip_y.get())
        for v_n in ['vinterp', 'v']:
            for w_n in ['winterp', 'w']:
                if v_n in self._ds.fields_3d and w_n in self._ds.fields_3d:
                    v = self._ds.get_field(v_n, self._t_idx)[::sy, ::sx, ii]
                    w = self._ds.get_field(w_n, self._t_idx)[::sy, ::sx, ii]
                    self._draw_wind(ax, yh[::sx], zh[::sy], v, w)
                    return

    # ── save PNG ─────────────────────────────────────────────────────────────

    def _save_png(self):
        if self._ds is None:
            messagebox.showwarning("No data", "Open a file first.")
            return
        field = self.v_field.get()
        t_sec = self._ds.times[self._t_idx]
        default = f"{field}_t{int(t_sec):06d}.png"
        path = filedialog.asksaveasfilename(
            initialfile=default,
            defaultextension='.png',
            filetypes=[("PNG", "*.png"), ("All", "*.*")],
            title="Save frame as PNG")
        if path:
            self._fig.savefig(path, dpi=150, bbox_inches='tight')
            self._gif_progress.config(text=f"Saved: {os.path.basename(path)}")

    def _copy_to_clipboard(self):
        try:
            _copy_figure_to_clipboard(self._fig)
            self._gif_progress.config(text="Copied!")
            self.after(2000, lambda: self._gif_progress.config(text=""))
        except Exception as e:
            messagebox.showerror("Clipboard error", str(e))

    # ── save GIF ─────────────────────────────────────────────────────────────

    def _save_gif(self):
        if self._ds is None:
            messagebox.showwarning("No data", "Open a file first.")
            return
        path = filedialog.asksaveasfilename(
            initialfile=f"{self.v_field.get()}.gif",
            defaultextension='.gif',
            filetypes=[("GIF", "*.gif"), ("All", "*.*")],
            title="Save animation as GIF")
        if not path:
            return
        threading.Thread(target=self._gif_worker, args=(path,),
                         daemon=True).start()

    def _gif_worker(self, path):
        try:
            t0 = float(self.v_gif_t0.get())
        except ValueError:
            t0 = self._ds.times[0]
        try:
            t1 = float(self.v_gif_t1.get())
        except ValueError:
            t1 = self._ds.times[-1]

        times = self._ds.times
        idx_range = [i for i, t in enumerate(times) if t0 <= t <= t1]
        if not idx_range:
            self.after(0, lambda: messagebox.showwarning(
                "GIF", "No time steps in the selected range."))
            return

        fps  = max(1, self.v_fps.get())
        n    = len(idx_range)
        self.after(0, lambda: self._gif_progress.config(
            text=f"Saving {n} frames…"))

        writer = PillowWriter(fps=fps)
        with writer.saving(self._fig, path, dpi=120):
            for k, ti in enumerate(idx_range):
                self._t_idx = ti
                self._plot()
                self._fig.canvas.draw()
                writer.grab_frame()
                msg = f"Frame {k+1}/{n}…"
                self.after(0, lambda m=msg: self._gif_progress.config(text=m))

        base = os.path.basename(path)
        self.after(0, lambda: self._gif_progress.config(
            text=f"Saved {n} frames → {base}"))


# ---------------------------------------------------------------------------
# Radar display window
# ---------------------------------------------------------------------------

# Weather colormaps via pyart (registers under 'pyart_*' names); fall back gracefully
try:
    import pyart  # noqa: F401  — side-effect: registers colormaps in matplotlib
    _CMAP_SPECTRAL = 'ChaseSpectral'
    _CMAP_VEL      = 'Carbone42'
except ImportError:
    _CMAP_SPECTRAL = 'gist_ncar'
    _CMAP_VEL      = 'RdBu_r'


class RadarWindow(tk.Toplevel):
    """Popup showing the latest radar scan.

    PPI: native polar projection axes (no Cartesian regridding).
    RHI: range–height axes using 2-D coordinate pcolormesh.
    """

    def __init__(self, master):
        super().__init__(master)
        self.title("Virtual Radar")
        self.minsize(620, 600)
        self._result    = None
        self._mode      = None
        self._radar_loc = None

        self._PCFG = {
            'refl': dict(label='Z  (dBZ)',      cmap=_CMAP_SPECTRAL, vmin=-10,   vmax=75),
            'vel':  dict(label='Vr  (m/s)',     cmap=_CMAP_VEL,      vmin=None,  vmax=None),
            'zdr':  dict(label='ZDR  (dB)',     cmap=_CMAP_SPECTRAL, vmin=-1,    vmax=8),
            'kdp':  dict(label='KDP  (°/km)',  cmap=_CMAP_SPECTRAL, vmin=0,     vmax=6),
            'cc':   dict(label='CC',            cmap=_CMAP_SPECTRAL, vmin=0.5,   vmax=1.0),
            'circ': dict(label='Circ  (m²/s)', cmap='RdBu_r',       vmin=-500,  vmax=500),
            'conv': dict(label='Conv  (s⁻¹)',  cmap='RdBu_r',       vmin=-0.02, vmax=0.02),
        }

        # Rescan state (populated by update_scan)
        self._ds          = None
        self._radar       = None
        self._scan_angle  = None
        self._t_idx       = 0
        self._scanning    = False
        self._pending_t   = None   # queued t_idx if slider moved during a scan
        self._rescan_af   = None   # debounce handle

        self._info_lbl = ttk.Label(self, text="", font=('Courier', 9))
        self._info_lbl.pack(fill='x', padx=6, pady=(4, 0))

        # Product selector
        self._current_product = 'refl'
        self._v_product = tk.StringVar(value='refl')
        _pr = ttk.Frame(self)
        _pr.pack(fill='x', padx=6, pady=(2, 0))
        for pname, plbl in [('refl', 'Z'), ('vel', 'Vr'), ('zdr', 'ZDR'),
                             ('kdp', 'KDP'), ('cc', 'CC'), ('circ', 'Circ'), ('conv', 'Conv')]:
            ttk.Radiobutton(_pr, text=plbl, variable=self._v_product, value=pname,
                            command=self._on_product_change).pack(side='left', padx=4)

        # Colorbar range controls
        _cr = ttk.Frame(self)
        _cr.pack(fill='x', padx=6, pady=(2, 0))
        ttk.Label(_cr, text="vmin:").pack(side='left')
        self._v_vmin = tk.StringVar(value='')
        self._v_vmax = tk.StringVar(value='')
        _emin = ttk.Entry(_cr, textvariable=self._v_vmin, width=7)
        _emin.pack(side='left', padx=(2, 6))
        ttk.Label(_cr, text="vmax:").pack(side='left')
        _emax = ttk.Entry(_cr, textvariable=self._v_vmax, width=7)
        _emax.pack(side='left', padx=(2, 6))
        ttk.Button(_cr, text="Apply", command=self._on_apply_limits).pack(side='left', padx=2)
        ttk.Button(_cr, text="Reset", command=self._on_reset_limits).pack(side='left', padx=2)
        self._v_symmetric = tk.BooleanVar(value=False)
        ttk.Checkbutton(_cr, text="Sym", variable=self._v_symmetric,
                        command=self._on_apply_limits).pack(side='left', padx=6)
        _emin.bind('<Return>', lambda _: self._on_apply_limits())
        _emax.bind('<Return>', lambda _: self._on_apply_limits())

        # Range zoom slider
        _rz = ttk.Frame(self)
        _rz.pack(fill='x', padx=6, pady=(2, 4))
        ttk.Label(_rz, text="Range:").pack(side='left')
        self._v_r_zoom = tk.DoubleVar(value=200.0)
        self._r_zoom_lbl = ttk.Label(_rz, text="200 km", width=8)
        self._r_zoom_slider = ttk.Scale(_rz, from_=5, to=200, orient='horizontal',
                                        variable=self._v_r_zoom,
                                        command=self._on_r_zoom)
        self._r_zoom_slider.pack(side='left', fill='x', expand=True, padx=4)
        self._r_zoom_lbl.pack(side='left')

        # Angle slider (elevation for PPI, azimuth for RHI)
        _af = ttk.Frame(self)
        _af.pack(fill='x', padx=6, pady=(0, 2))
        self._angle_lbl_name = ttk.Label(_af, text="El (°):", width=7)
        self._angle_lbl_name.pack(side='left')
        self._v_angle = tk.DoubleVar(value=0.5)
        self._angle_val_lbl = ttk.Label(_af, text=" 0.5°", width=7)
        self._angle_slider = ttk.Scale(_af, from_=0.0, to=20.0, orient='horizontal',
                                       variable=self._v_angle,
                                       command=self._on_angle_change)
        self._angle_slider.pack(side='left', fill='x', expand=True, padx=4)
        self._angle_val_lbl.pack(side='left')

        # Time navigation row
        _tf = ttk.Frame(self)
        _tf.pack(fill='x', padx=6, pady=(0, 4))
        ttk.Button(_tf, text="◀◀", width=3, command=self._t_first).pack(side='left', padx=2)
        ttk.Button(_tf, text="◀",  width=3, command=self._t_prev ).pack(side='left', padx=2)
        ttk.Button(_tf, text="▶",  width=3, command=self._t_next ).pack(side='left', padx=2)
        ttk.Button(_tf, text="▶▶", width=3, command=self._t_last ).pack(side='left', padx=2)
        self._t_slider = ttk.Scale(_tf, from_=0, to=0, orient='horizontal',
                                   command=self._on_t_change)
        self._t_slider.pack(side='left', fill='x', expand=True, padx=4)
        self._t_lbl  = ttk.Label(_tf, text="t = —", width=12,
                                  font=('Courier', 10, 'bold'))
        self._t_lbl.pack(side='left', padx=4)
        self._t_busy = ttk.Label(_tf, text="", foreground='#226')
        self._t_busy.pack(side='left')

        self._fig    = Figure(figsize=(6.2, 5.4), dpi=100)
        self._ax     = None
        self._cax    = None
        self._cbar   = None
        self._canvas = FigureCanvasTkAgg(self._fig, master=self)
        self._canvas.get_tk_widget().pack(fill='both', expand=True)
        _tb_row = ttk.Frame(self)
        _tb_row.pack(fill='x')
        NavigationToolbar2Tk(self._canvas, _tb_row, pack_toolbar=False).pack(side='left', fill='x', expand=True)
        ttk.Button(_tb_row, text="Copy to Clipboard",
                   command=self._copy_to_clipboard).pack(side='right', padx=6, pady=2)

    # ── public API ─────────────────────────────────────────────────────────────

    def update_scan(self, result, mode, product, radar_loc,
                    ds=None, radar=None, scan_angle=None, t_idx=0):
        self._result    = result
        self._mode      = mode
        self._radar_loc = radar_loc
        if ds is not None:
            self._ds         = ds
            self._radar      = radar
            self._scan_angle = scan_angle
        self._t_idx = t_idx
        # Sync angle slider to current mode
        if mode == 'ppi':
            self._angle_lbl_name.config(text="El (°):")
            self._angle_slider.configure(from_=0.1, to=20.0)
        else:
            self._angle_lbl_name.config(text="Az (°):")
            self._angle_slider.configure(from_=0.0, to=360.0)
        if scan_angle is not None:
            self._v_angle.set(scan_angle)
            self._angle_val_lbl.config(text=f"{scan_angle:5.1f}°")
        # Sync time slider
        if self._ds is not None:
            nt = self._ds.ntimes
            self._t_slider.configure(to=max(0, nt - 1))
            self._t_slider.set(t_idx)
            self._t_lbl.config(text=f"t = {self._ds.times[t_idx]:.0f} s")
        # Set range slider to match this scan's maximum range
        r_edges  = result.get('r_edges')
        r_km_arr = result.get('r_km')
        if r_edges is not None:
            r_max = float(r_edges[-1])
        elif r_km_arr is not None:
            r_max = float(r_km_arr[-1])
        else:
            r_max = 200.0
        self._r_zoom_slider.configure(to=max(r_max, 5.0))
        self._v_r_zoom.set(r_max)
        self._r_zoom_lbl.config(text=f"{r_max:.0f} km")
        self.replot(product)
        self.lift()

    def replot(self, product):
        if self._result is None:
            return
        result = self._result
        mode   = self._mode
        loc    = self._radar_loc
        pcfg   = self._PCFG.get(product, self._PCFG['refl'])

        data = result.get(product)
        if data is None:
            return

        # Track product; clear manual limits and sync radio button on change
        if product != self._current_product:
            self._current_product = product
            self._v_product.set(product)
            self._v_vmin.set('')
            self._v_vmax.set('')

        is_ppi = (mode == 'ppi')
        self._reset_figure(is_ppi)

        # Default limits
        vmin, vmax = pcfg['vmin'], pcfg['vmax']
        if product == 'vel':
            vm   = float(result.get('v_max', 30.0))
            vmin, vmax = -vm, vm
        # Override with user entries if set
        try:    vmin = float(self._v_vmin.get())
        except ValueError: pass
        try:    vmax = float(self._v_vmax.get())
        except ValueError: pass
        # Symmetrize: |vmin| == |vmax| == max of the two
        if self._v_symmetric.get() and vmin is not None and vmax is not None:
            val  = max(abs(vmin), abs(vmax))
            vmin, vmax = -val, val
        norm = Normalize(vmin=vmin, vmax=vmax)

        finite = data[np.isfinite(data)]
        if len(finite) == 0:
            self._ax.text(0.5, 0.5, "No signal above noise floor",
                          ha='center', va='center',
                          transform=self._ax.transAxes, color='gray')
            self._canvas.draw_idle()
            return

        im = self._draw_ppi(result, data, norm, pcfg, loc) if is_ppi \
             else self._draw_rhi(result, data, norm, pcfg, loc)

        if im is not None:
            if self._cbar is None:
                self._cbar = self._fig.colorbar(im, cax=self._cax)
            else:
                self._cbar.update_normal(im)
            self._cbar.set_label(pcfg['label'], fontsize=9)

        self._canvas.draw_idle()

    def _on_product_change(self):
        self.replot(self._v_product.get())

    def _on_apply_limits(self):
        self.replot(self._current_product)

    def _on_reset_limits(self):
        self._v_vmin.set('')
        self._v_vmax.set('')
        self.replot(self._current_product)

    def _copy_to_clipboard(self):
        try:
            _copy_figure_to_clipboard(self._fig)
            self._t_busy.config(text="Copied!")
            self.after(2000, lambda: self._t_busy.config(text=""))
        except Exception as e:
            messagebox.showerror("Clipboard error", str(e))

    def _on_r_zoom(self, _=None):
        km = round(self._v_r_zoom.get())
        self._r_zoom_lbl.config(text=f"{km} km")
        self.replot(self._current_product)

    def _on_angle_change(self, _=None):
        if self._ds is None:
            return
        angle = round(self._v_angle.get(), 1)
        self._angle_val_lbl.config(text=f"{angle:5.1f}°")
        if self._rescan_af is not None:
            self.after_cancel(self._rescan_af)
        self._rescan_af = self.after(250, lambda: self._rescan(self._t_idx))

    def _on_t_change(self, _=None):
        if self._ds is None:
            return
        t_idx = int(round(float(self._t_slider.get())))
        self._t_lbl.config(text=f"t = {self._ds.times[t_idx]:.0f} s")
        if self._rescan_af is not None:
            self.after_cancel(self._rescan_af)
        self._rescan_af = self.after(250, lambda: self._rescan(t_idx))

    def _t_first(self):
        if self._ds is None:
            return
        self._t_slider.set(0)
        self._rescan(0)

    def _t_last(self):
        if self._ds is None:
            return
        n = self._ds.ntimes - 1
        self._t_slider.set(n)
        self._rescan(n)

    def _t_prev(self):
        if self._ds is None:
            return
        t = max(0, self._t_idx - 1)
        self._t_slider.set(t)
        self._rescan(t)

    def _t_next(self):
        if self._ds is None:
            return
        t = min(self._ds.ntimes - 1, self._t_idx + 1)
        self._t_slider.set(t)
        self._rescan(t)

    def _rescan(self, t_idx):
        if self._ds is None or self._radar is None:
            return
        if self._scanning:
            self._pending_t = t_idx
            return
        self._scanning  = True
        self._pending_t = None
        self._t_busy.config(text="Scanning…")
        mode  = self._mode
        radar = self._radar
        ds    = self._ds
        angle = round(self._v_angle.get(), 1)

        import threading
        def _run():
            try:
                if mode == 'ppi':
                    result = radar.scan_ppi(ds, t_idx, angle)
                else:
                    result = radar.scan_rhi(ds, t_idx, angle)
                def _done():
                    if not self.winfo_exists():
                        return
                    self._result = result
                    self._t_idx  = t_idx
                    r_edges  = result.get('r_edges')
                    r_km_arr = result.get('r_km')
                    r_max = float(r_edges[-1]) if r_edges is not None \
                            else (float(r_km_arr[-1]) if r_km_arr is not None else 200.0)
                    self._r_zoom_slider.configure(to=max(r_max, 5.0))
                    self.replot(self._current_product)
                    self._t_busy.config(text="")
                    self._scanning = False
                    if self._pending_t is not None:
                        pending, self._pending_t = self._pending_t, None
                        self._rescan(pending)
                self.after(0, _done)
            except Exception as exc:
                def _err(msg=str(exc)):
                    if not self.winfo_exists():
                        return
                    self._t_busy.config(text=f"Error: {msg}")
                    self._scanning = False
                self.after(0, _err)

        threading.Thread(target=_run, daemon=True).start()

    # ── private helpers ────────────────────────────────────────────────────────

    def _reset_figure(self, polar):
        self._fig.clf()
        self._cbar = None
        if polar:
            self._ax  = self._fig.add_axes([0.06, 0.05, 0.80, 0.88],
                                            projection='polar')
        else:
            self._ax  = self._fig.add_axes([0.10, 0.08, 0.74, 0.84])
        self._cax = self._fig.add_axes([0.89, 0.08, 0.025, 0.84])

    def _draw_ppi(self, result, data, norm, pcfg, loc):
        # az_edges: (n_az+1,) radians, meteorological convention (0=N, CW)
        # r_edges:  (n_gates+1,) km
        # data:     (n_az, n_gates)
        az_edges = result['az_edges']
        r_edges  = result['r_edges']
        ax = self._ax
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        # pcolormesh(X, Y, C) expects C.shape == (len(Y)-1, len(X)-1)
        # X=az (cols), Y=r (rows) → need data transposed to (n_gates, n_az)
        im = ax.pcolormesh(az_edges, r_edges, data.T,
                           cmap=pcfg['cmap'], norm=norm, shading='flat')
        r_zoom = min(round(self._v_r_zoom.get()), float(r_edges[-1]))
        ax.set_rmax(r_zoom)
        el      = result.get('el_deg', 0.0)
        vm      = result.get('v_max', 30.0)
        loc_str = f"({loc[0]:.1f}, {loc[1]:.1f}) km" if loc else "?"
        ax.set_title(f"PPI  El={el:.1f}°   Radar @ {loc_str}", pad=14, fontsize=10)
        self._info_lbl.config(
            text=f"PPI  El={el:.1f}°   Vmax=±{vm:.0f} m/s   Rmax={r_zoom:.0f} km")
        return im

    def _draw_rhi(self, result, data, norm, pcfg, loc):
        r_km = result['r_km']           # (n_gates,)
        h_km = result['h_km'] / 1000.0  # (n_el, n_gates) m → km
        n_el, n_g = data.shape
        ax = self._ax

        # Range edges (1-D)
        if n_g > 1:
            r_e = np.empty(n_g + 1, dtype=np.float32)
            r_e[0]    = r_km[0] - (r_km[1] - r_km[0]) / 2.0
            r_e[1:-1] = (r_km[:-1] + r_km[1:]) / 2.0
            r_e[-1]   = r_km[-1] + (r_km[-1] - r_km[-2]) / 2.0
        else:
            r_e = np.array([0.0, float(r_km[0])], dtype=np.float32)

        # Height edges (n_el+1, n_g) — one edge per elevation boundary
        if n_el > 1:
            dh  = np.diff(h_km, axis=0)
            h_e = np.vstack([h_km[0:1]  - dh[0:1]  / 2.0,
                             (h_km[:-1] + h_km[1:]) / 2.0,
                              h_km[-1:] + dh[-1:]  / 2.0])
        else:
            h_e = np.vstack([np.zeros((1, n_g), dtype=np.float32), h_km])

        # Expand height edges to (n_el+1, n_g+1) for pcolormesh
        h_e2             = np.empty((n_el + 1, n_g + 1), dtype=np.float32)
        h_e2[:, 0]       = h_e[:, 0]
        h_e2[:, 1:-1]    = (h_e[:, :-1] + h_e[:, 1:]) / 2.0
        h_e2[:, -1]      = h_e[:, -1]
        r_e2             = np.tile(r_e, (n_el + 1, 1))   # (n_el+1, n_g+1)

        im = ax.pcolormesh(r_e2, h_e2, data,
                           cmap=pcfg['cmap'], norm=norm, shading='flat')
        ax.set_xlabel('Range  (km)')
        ax.set_ylabel('Height  (km)')
        r_zoom = min(round(self._v_r_zoom.get()), float(r_km[-1]))
        ax.set_xlim(0, r_zoom)
        # Cap y-axis using reflectivity extent so all products share the same ylim
        refl  = result.get('refl', data)
        valid = np.isfinite(refl)
        # Restrict height max to gates within the zoomed range
        in_range = r_km <= r_zoom
        refl_in  = refl[:, in_range] if refl.ndim == 2 else refl
        valid_in = np.isfinite(refl_in)
        h_in     = h_km[:, in_range] if h_km.ndim == 2 else h_km
        h_top = float(h_in[valid_in].max()) * 1.1 if valid_in.any() else float(h_km.max())
        ax.set_ylim(0, h_top)
        az      = result.get('az_deg', 0.0)
        vm      = result.get('v_max', 30.0)
        loc_str = f"({loc[0]:.1f}, {loc[1]:.1f}) km" if loc else "?"
        ax.set_title(f"RHI  Az={az:.1f}°   Radar @ {loc_str}", fontsize=10)
        self._info_lbl.config(
            text=f"RHI  Az={az:.1f}°   Vmax=±{vm:.0f} m/s   Rmax={r_zoom:.0f} km")
        return im


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    app = CM1Viewer()
    app.protocol('WM_DELETE_WINDOW', lambda: (app._stop_watching(), app.destroy()))
    app.mainloop()
